from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


logger = logging.getLogger(__name__)


PROTECTED_MEDICAL_FEATURES = {
    # Baseline FHR
    "LB",
    # Variability
    "MSTV", "MLTV", "ASTV", "ALTV",
    # Decelerations
    "DL", "DS", "DP",
}


@dataclass
class SelectionResult:
    selected_features: List[str]
    scores: pd.DataFrame


class FeatureSelector:
    """Feature selection with multiple criteria and medical safeguards.

    Methods combine: chi-square (discrete relevance), ANOVA F (linear separability),
    mutual information (nonlinear dependency), and L1-logistic sparsity. Key medically
    important features are always retained.
    """

    def __init__(self, target_column: str = "NSP", protected: Optional[List[str]] = None) -> None:
        # Normalize target name to canonical form
        self.target_column = target_column.strip()
        self.protected = set(PROTECTED_MEDICAL_FEATURES if protected is None else protected)
        # Case-insensitive canonicalization for columns
        self.alias_map: Dict[str, str] = {k.lower(): k for k in PROTECTED_MEDICAL_FEATURES | {self.target_column}}

    # ---------- scoring ----------
    def _prepare_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        # Normalize columns to match expected casing
        rename = {}
        for c in df.columns:
            key = str(c).strip().lower()
            if key == self.target_column.lower():
                rename[c] = self.target_column
        if rename:
            df = df.rename(columns=rename)

        if self.target_column not in df.columns:
            raise KeyError(f"Target column '{self.target_column}' not found")
        y = pd.to_numeric(df[self.target_column], errors="coerce")
        X = df.drop(columns=[self.target_column])
        # Drop rows with NaN target before casting to int
        mask = y.notna()
        dropped = int((~mask).sum())
        if dropped:
            logger.info("Feature selection: dropped %d rows with NaN target '%s'", dropped, self.target_column)
        y = y.loc[mask].astype(int)
        X = X.loc[mask]
        # keep only numeric columns for stats
        X = X.select_dtypes(include=[np.number]).copy()
        # Fill NaNs minimally to allow tests
        X = X.fillna(X.median(numeric_only=True))
        return X, y

    def _score_chi2(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        scaler = MinMaxScaler()
        X_pos = scaler.fit_transform(X.clip(lower=0))
        scores, _ = chi2(X_pos, y)
        return pd.Series(scores, index=X.columns, name="chi2")

    def _score_anova(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        scores, _ = f_classif(X, y)
        return pd.Series(scores, index=X.columns, name="anova_f")

    def _score_mi(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        scores = mutual_info_classif(X, y, discrete_features=False, random_state=42)
        return pd.Series(scores, index=X.columns, name="mutual_info")

    def _score_l1(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        # Use multinomial logistic with L1; scale features to comparable ranges
        scaler = MinMaxScaler()
        Xs = scaler.fit_transform(X)
        try:
            clf = LogisticRegression(penalty="l1", solver="saga", multi_class="multinomial", max_iter=500, C=1.0, random_state=42)
            clf.fit(Xs, y)
            # Importance as mean absolute coefficient across classes
            coef = np.mean(np.abs(clf.coef_), axis=0)
            return pd.Series(coef, index=X.columns, name="l1_coef")
        except Exception as exc:
            logger.warning("L1 selection failed: %s", exc)
            return pd.Series(0.0, index=X.columns, name="l1_coef")

    def get_feature_importance_report(self, df: pd.DataFrame) -> pd.DataFrame:
        X, y = self._prepare_xy(df)
        reports = [
            self._score_chi2(X, y),
            self._score_anova(X, y),
            self._score_mi(X, y),
            self._score_l1(X, y),
        ]
        scores = pd.concat(reports, axis=1)
        # Normalize each score column to [0,1] for comparability
        for col in scores.columns:
            col_min = scores[col].min()
            col_max = scores[col].max()
            rng = col_max - col_min
            if rng <= 0:
                scores[col] = 0.0
            else:
                scores[col] = (scores[col] - col_min) / rng
        scores["aggregate_score"] = scores.mean(axis=1)
        scores = scores.sort_values("aggregate_score", ascending=False)
        return scores

    def validate_medical_relevance(self, features: List[str]) -> List[str]:
        # Ensure protected features are present
        output = list(dict.fromkeys(list(self.protected) + features))
        return output

    def select_top_k_features(self, df: pd.DataFrame, k: int = 20) -> SelectionResult:
        scores = self.get_feature_importance_report(df)
        top = list(scores.index[:max(k, 0)])
        selected = self.validate_medical_relevance(top)
        logger.info("Selected %d features (k=%d + protected %d)", len(selected), k, len(self.protected))
        return SelectionResult(selected_features=selected, scores=scores)

    # ---------- Visualization ----------
    def plot_importance(self, scores: pd.DataFrame, top_n: int = 20, outfile: Optional[Path] = None) -> None:
        import matplotlib.pyplot as plt  # local import to avoid headless issues
        import seaborn as sns
        top_scores = scores.head(top_n).iloc[::-1]
        plt.figure(figsize=(8, max(4, top_n * 0.35)))
        sns.barplot(x=top_scores["aggregate_score"], y=top_scores.index, orient="h")
        plt.xlabel("Aggregate importance (normalized)")
        plt.ylabel("Feature")
        plt.title("Feature importance (aggregate)")
        plt.tight_layout()
        if outfile is not None:
            plt.savefig(outfile, dpi=200)
            logger.info("Saved feature importance plot to %s", outfile)
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # Example manual run
    # from src.data.data_loader import DataLoader
    # df = DataLoader().clean_ctg_dataframe(DataLoader().load(sheet="Data"))
    # fs = FeatureSelector()
    # result = fs.select_top_k_features(df, k=20)
    # result.scores.to_csv("reports/feature_selection_scores.csv")
    # fs.plot_importance(result.scores, outfile=Path("reports/feature_importance.png"))


