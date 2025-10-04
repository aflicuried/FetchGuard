from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, GroupKFold, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import mlflow
import mlflow.sklearn


logger = logging.getLogger(__name__)


def load_engineered_or_clean(
    fe_path: Path = Path("data/processed/ctg_with_features.csv"),
    clean_path: Path = Path("data/processed/ctg_clean.csv"),
) -> pd.DataFrame:
    if fe_path.exists():
        return pd.read_csv(fe_path)
    if clean_path.exists():
        return pd.read_csv(clean_path)
    raise FileNotFoundError("No processed dataset found. Run scripts/preprocess.py first.")


def get_features_and_target(df: pd.DataFrame, target: str = "NSP") -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataset")
    y = pd.to_numeric(df[target], errors="coerce")
    mask = y.notna()
    if (~mask).any():
        logger.info("Dropping %d rows with NaN target", int((~mask).sum()))
    y = y.loc[mask].astype(int)
    X = df.loc[mask].select_dtypes(include=[np.number]).drop(columns=[col for col in [target] if col in df], errors="ignore")
    return X, y


def build_classifier(name: str, class_weight: Optional[str] = None, random_state: int = 42):
    name = name.lower()
    if name == "logreg":
        return LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            multi_class="auto",
            class_weight=class_weight,
            max_iter=200,
            random_state=random_state,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            random_state=random_state,
            n_jobs=-1,
            class_weight=class_weight or "balanced_subsample",
        )
    if name == "xgb":
        return XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
    if name == "lgbm":
        return LGBMClassifier(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multiclass",
            class_weight=None,
            random_state=random_state,
            n_jobs=-1,
        )
    if name == "svm":
        return SVC(kernel="rbf", probability=True, class_weight=class_weight or "balanced", random_state=random_state)
    raise ValueError(f"Unknown algorithm: {name}")


def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    algo: str,
    use_smote: bool,
    class_weight: Optional[str],
    n_splits: int,
    random_state: int,
    groups: Optional[pd.Series] = None,
    time_series_split: bool = False,
) -> Dict[str, float]:
    # Build pipeline
    clf = build_classifier(algo, class_weight=class_weight, random_state=random_state)
    steps: List[Tuple[str, object]] = [("scaler", StandardScaler(with_mean=False))]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("clf", clf))
    pipe = ImbPipeline(steps=steps)

    # Choose splitter
    if groups is not None:
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(X, y, groups=groups)
    elif time_series_split:
        splitter = TimeSeriesSplit(n_splits=n_splits)
        splits = splitter.split(X, y)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = splitter.split(X, y)

    metrics: Dict[str, List[float]] = {"acc": [], "bacc": [], "f1_macro": [], "roc_auc_ovr": []}
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        bacc = balanced_accuracy_score(y_te, y_pred)
        f1m = f1_score(y_te, y_pred, average="macro")
        # ROC AUC (if proba available)
        try:
            y_proba = pipe.predict_proba(X_te)
            roc = roc_auc_score(y_te, y_proba, multi_class="ovr")
        except Exception:
            roc = np.nan

        metrics["acc"].append(acc)
        metrics["bacc"].append(bacc)
        metrics["f1_macro"].append(f1m)
        metrics["roc_auc_ovr"].append(roc)

        logger.info("Fold %d: acc=%.3f bacc=%.3f f1_macro=%.3f roc=%.3f", fold, acc, bacc, f1m, roc)

    summary = {k: float(np.nanmean(v)) for k, v in metrics.items()}
    return summary


def per_doctor_validation(X: pd.DataFrame, y: pd.Series, doctor: pd.Series, algo: str, use_smote: bool, class_weight: Optional[str], random_state: int) -> Dict[str, float]:
    clf = build_classifier(algo, class_weight=class_weight, random_state=random_state)
    steps: List[Tuple[str, object]] = [("scaler", StandardScaler(with_mean=False))]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("clf", clf))
    pipe = ImbPipeline(steps=steps)

    doctors = pd.unique(doctor.dropna())
    accs = []
    for doc in doctors:
        mask_te = (doctor == doc)
        mask_tr = ~mask_te
        if mask_te.sum() == 0 or mask_tr.sum() == 0:
            continue
        pipe.fit(X.loc[mask_tr], y.loc[mask_tr])
        y_pred = pipe.predict(X.loc[mask_te])
        accs.append(accuracy_score(y.loc[mask_te], y_pred))
    return {"per_doctor_acc_mean": float(np.mean(accs)) if accs else np.nan}


def log_model_artifacts(run_name: str, model, algo: str, X_cols: List[str], scores: Dict[str, float], model_out: Path) -> None:
    mlflow.set_tag("algo", algo)
    for k, v in scores.items():
        mlflow.log_metric(k, v)
    # Save model
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    mlflow.log_artifact(str(model_out), artifact_path="model")
    # Feature importance if available
    try:
        if hasattr(model.named_steps["clf"], "feature_importances_"):
            importances = model.named_steps["clf"].feature_importances_
            imp_df = pd.DataFrame({"feature": X_cols, "importance": importances}).sort_values("importance", ascending=False)
            tmp = model_out.parent / f"feature_importance_{algo}.csv"
            imp_df.to_csv(tmp, index=False)
            mlflow.log_artifact(str(tmp), artifact_path="importance")
    except Exception:
        pass


def train_main():
    parser = argparse.ArgumentParser(description="CTG Training Pipeline")
    parser.add_argument("--algo", type=str, default="rf", choices=["logreg", "rf", "xgb", "lgbm", "svm"]) 
    parser.add_argument("--use-smote", action="store_true", help="Enable SMOTE oversampling")
    parser.add_argument("--class-weight", type=str, default=None, choices=[None, "balanced", "balanced_subsample"], help="Class weight strategy")
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds (stratified)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--doctor-col", type=str, default=None, help="Column name for obstetrician ID for per-doctor validation")
    parser.add_argument("--time-col", type=str, default=None, help="Column name for timestamp to use time-series split")
    parser.add_argument("--target", type=str, default="NSP")
    parser.add_argument("--mlflow-uri", type=str, default=None)
    parser.add_argument("--mlflow-exp", type=str, default="CTG-Experiments")
    parser.add_argument("--model-out", type=Path, default=Path("models/saved/model.joblib"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_exp)

    df = load_engineered_or_clean()
    X, y = get_features_and_target(df, target=args.target)
    y = y - 1

    groups = None
    if args.doctor_col and args.doctor_col in df.columns:
        groups = df.loc[y.index, args.doctor_col]

    with mlflow.start_run(run_name=f"{args.algo}-cv{args.cv}"):
        mlflow.log_params({
            "algo": args.algo,
            "use_smote": args.use_smote,
            "class_weight": args.class_weight,
            "cv": args.cv,
            "random_state": args.random_state,
            "target": args.target,
        })

        # Cross-validation
        time_split = args.time_col is not None and args.time_col in df.columns
        scores = run_cv(
            X, y, args.algo, args.use_smote, args.class_weight, args.cv, args.random_state, groups=groups, time_series_split=time_split
        )

        # Fit on full data for artifact
        clf = build_classifier(args.algo, class_weight=args.class_weight, random_state=args.random_state)
        steps: List[Tuple[str, object]] = [("scaler", StandardScaler(with_mean=False))]
        if args.use_smote:
            steps.append(("smote", SMOTE(random_state=args.random_state)))
        steps.append(("clf", clf))
        model = ImbPipeline(steps=steps)
        model.fit(X, y)

        # Optional per-doctor validation
        if groups is not None:
            pd_scores = per_doctor_validation(X, y, groups, args.algo, args.use_smote, args.class_weight, args.random_state)
            scores.update(pd_scores)

        log_model_artifacts("model", model, args.algo, list(X.columns), scores, args.model_out)

        logger.info("CV summary: %s", scores)


if __name__ == "__main__":
    train_main()


