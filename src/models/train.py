"""
CTG 胎儿心电图分类训练脚本

该脚本实现了完整的机器学习训练流程，支持多种分类器和交叉验证策略。
主要功能：
- 支持多种分类算法（Logistic Regression, Random Forest, XGBoost, LightGBM, SVM）
- 灵活的数据加载（标准化数据或原始清洗数据）
- 交叉验证评估（StratifiedKFold, GroupKFold, TimeSeriesSplit）
- 类别不平衡处理（SMOTE, class_weight）
- MLflow 实验跟踪和模型管理
- 完整的医学评估指标（敏感度、特异度、PPV、NPV等）
- SHAP模型可解释性分析
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    roc_curve,
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

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not installed. Install with: pip install shap")


logger = logging.getLogger(__name__)

# Medical class definitions
CLASS_NAMES = {0: "Normal", 1: "Suspect", 2: "Pathological"}
CLASS_RISK_LEVELS = {0: "Low", 1: "Moderate", 2: "High"}

# Cost matrix for medical decision making
DEFAULT_COST_MATRIX = np.array([
    [0,   1,   5],   # True Normal: misclassify as Suspect(1) or Pathological(5)
    [1,   0,   3],   # True Suspect: misclassify as Normal(1) or Pathological(3)
    [20,  10,  0]    # True Pathological: NEVER miss (20 for Normal, 10 for Suspect)
])


def load_data(
    use_normalized: bool = True,
    normalized_path: Path = Path("data/processed/ctg_normalized.csv"),
    clean_path: Path = Path("data/processed/ctg_clean.csv"),
) -> pd.DataFrame:
    """
    加载CTG数据集

    Args:
        use_normalized: 是否使用标准化后的数据（默认True）
        normalized_path: 标准化数据文件路径
        clean_path: 原始清洗数据文件路径

    Returns:
        pd.DataFrame: 加载的数据集

    Raises:
        FileNotFoundError: 当指定的数据文件不存在时
    """
    if use_normalized:
        if normalized_path.exists():
            logger.info(f"Loading normalized data from {normalized_path}")
            return pd.read_csv(normalized_path)
        else:
            raise FileNotFoundError(f"Normalized data not found at {normalized_path}")
    else:
        if clean_path.exists():
            logger.info(f"Loading clean data from {clean_path}")
            return pd.read_csv(clean_path)
        else:
            raise FileNotFoundError(f"Clean data not found at {clean_path}")


def get_features_and_target(df: pd.DataFrame, target: str = "NSP") -> Tuple[pd.DataFrame, pd.Series]:
    """
    从数据集中分离特征和目标变量

    Args:
        df: 输入数据集
        target: 目标列名（默认为"NSP"）

    Returns:
        Tuple[pd.DataFrame, pd.Series]: 特征矩阵X和目标向量y

    Raises:
        KeyError: 当目标列不存在时
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataset")

    # 转换目标列为数值类型
    y = pd.to_numeric(df[target], errors="coerce")
    mask = y.notna()
    if (~mask).any():
        logger.info("Dropping %d rows with NaN target", int((~mask).sum()))
    y = y.loc[mask].astype(int)

    # 排除所有标签列（NSP 和 CLASS）
    exclude_cols = [target, 'CLASS']
    X = df.loc[mask].select_dtypes(include=[np.number]).drop(
        columns=[col for col in exclude_cols if col in df],
        errors="ignore"
    )

    # 打印使用的特征，便于验证
    logger.info(f"Number of features: {X.shape[1]}")
    logger.info(f"CLASS excluded: {'CLASS' not in X.columns}")

    return X, y


def build_classifier(name: str, class_weight: Optional[str] = None, random_state: int = 42):
    """
    构建分类器实例

    Args:
        name: 算法名称（logreg/rf/xgb/lgbm/svm）
        class_weight: 类别权重策略
        random_state: 随机种子

    Returns:
        分类器实例

    Raises:
        ValueError: 当算法名称不支持时
    """
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
        # XGBoost不支持class_weight参数，使用scale_pos_weight或sample_weight
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
    """
    执行交叉验证评估

    Args:
        X: 特征矩阵
        y: 目标向量
        algo: 算法名称
        use_smote: 是否使用SMOTE过采样
        class_weight: 类别权重策略
        n_splits: 交叉验证折数
        random_state: 随机种子
        groups: 分组信息（用于GroupKFold）
        time_series_split: 是否使用时间序列分割

    Returns:
        Dict[str, float]: 包含各项指标均值的字典
    """
    # 构建pipeline
    clf = build_classifier(algo, class_weight=class_weight, random_state=random_state)
    steps: List[Tuple[str, object]] = [("scaler", StandardScaler(with_mean=False))]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("clf", clf))
    pipe = ImbPipeline(steps=steps)

    # 选择分割策略
    if groups is not None:
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(X, y, groups=groups)
    elif time_series_split:
        splitter = TimeSeriesSplit(n_splits=n_splits)
        splits = splitter.split(X, y)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = splitter.split(X, y)

    # 在每个fold上进行评估，同时收集所有预测结果
    metrics: Dict[str, List[float]] = {"acc": [], "bacc": [], "f1_macro": [], "roc_auc_ovr": []}
    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        # 为XGBoost计算sample_weight（如果需要类别平衡）
        if algo.lower() == 'xgb' and class_weight:
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight('balanced', y_tr)
            pipe.fit(X_tr, y_tr, clf__sample_weight=sample_weights)
        else:
            pipe.fit(X_tr, y_tr)

        y_pred = pipe.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        bacc = balanced_accuracy_score(y_te, y_pred)
        f1m = f1_score(y_te, y_pred, average="macro")

        # 计算ROC AUC（如果分类器支持概率预测）
        try:
            y_proba = pipe.predict_proba(X_te)
            roc = roc_auc_score(y_te, y_proba, multi_class="ovr")
        except Exception:
            y_proba = None
            roc = np.nan

        metrics["acc"].append(acc)
        metrics["bacc"].append(bacc)
        metrics["f1_macro"].append(f1m)
        metrics["roc_auc_ovr"].append(roc)

        # 收集预测结果用于详细医学评估
        all_y_true.extend(y_te.tolist())
        all_y_pred.extend(y_pred.tolist())
        if y_proba is not None:
            all_y_proba.extend(y_proba.tolist())

        logger.info("Fold %d: acc=%.3f bacc=%.3f f1_macro=%.3f roc=%.3f", fold, acc, bacc, f1m, roc)

    # 计算各指标的均值
    summary = {k: float(np.nanmean(v)) for k, v in metrics.items()}

    # 添加收集的预测结果到返回字典
    summary['cv_predictions'] = {
        'y_true': np.array(all_y_true),
        'y_pred': np.array(all_y_pred),
        'y_proba': np.array(all_y_proba) if all_y_proba else None
    }

    return summary


def per_doctor_validation(X: pd.DataFrame, y: pd.Series, doctor: pd.Series, algo: str, use_smote: bool, class_weight: Optional[str], random_state: int) -> Dict[str, float]:
    """
    执行按医生分组的交叉验证（留一医生验证）

    Args:
        X: 特征矩阵
        y: 目标向量
        doctor: 医生ID列
        algo: 算法名称
        use_smote: 是否使用SMOTE
        class_weight: 类别权重策略
        random_state: 随机种子

    Returns:
        Dict[str, float]: 包含按医生验证准确率的字典
    """
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


def save_training_summary(output_dir: Path, params: Dict, scores: Dict[str, float], timestamp: str) -> None:
    """
    将训练参数和结果保存到统一的JSON文件中

    Args:
        output_dir: 输出目录
        params: 训练参数字典
        scores: 评估指标字典
        timestamp: 时间戳字符串
    """
    summary = {
        "timestamp": timestamp,
        "parameters": params,
        "metrics": scores
    }

    summary_path = output_dir / "training_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Training summary saved to {summary_path}")


def log_model_artifacts(run_name: str, model, algo: str, X_cols: List[str], scores: Dict[str, float], model_out: Path, params: Dict, timestamp: str) -> None:
    """
    记录模型产物到MLflow和本地文件系统

    Args:
        run_name: 运行名称
        model: 训练好的模型
        algo: 算法名称
        X_cols: 特征列名列表
        scores: 评估指标字典
        model_out: 模型输出路径
        params: 训练参数字典
        timestamp: 时间戳字符串
    """
    # 记录MLflow标签和指标
    mlflow.set_tag("algo", algo)
    for k, v in scores.items():
        mlflow.log_metric(k, v)

    # 保存模型
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    mlflow.log_artifact(str(model_out), artifact_path="model")

    # 保存训练摘要（参数+指标合并到一个JSON文件）
    save_training_summary(model_out.parent, params, scores, timestamp)
    mlflow.log_artifact(str(model_out.parent / "training_summary.json"), artifact_path="summary")

    # 特征重要性（如果可用）
    try:
        if hasattr(model.named_steps["clf"], "feature_importances_"):
            importances = model.named_steps["clf"].feature_importances_
            imp_df = pd.DataFrame({"feature": X_cols, "importance": importances}).sort_values("importance", ascending=False)
            tmp = model_out.parent / "feature_importance.csv"
            imp_df.to_csv(tmp, index=False)
            mlflow.log_artifact(str(tmp), artifact_path="importance")
    except Exception:
        pass


def calculate_medical_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    计算详细的医学评估指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_proba: 预测概率（可选）

    Returns:
        包含所有医学指标的字典
    """
    metrics = {}

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    metrics['confusion_matrix'] = cm

    # 为每个类别计算指标
    for class_label in [0, 1, 2]:
        y_true_binary = (y_true == class_label).astype(int)
        y_pred_binary = (y_pred == class_label).astype(int)

        cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
        if cm_binary.shape == (2, 2):
            tn, fp, fn, tp = cm_binary.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

        # 计算医学指标
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        class_name = CLASS_NAMES[class_label]
        metrics[f'{class_name}_sensitivity'] = sensitivity
        metrics[f'{class_name}_specificity'] = specificity
        metrics[f'{class_name}_ppv'] = ppv
        metrics[f'{class_name}_npv'] = npv

    # 总体指标
    metrics['overall_accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

    # 成本敏感指标
    total_cost = np.sum(cm * DEFAULT_COST_MATRIX)
    metrics['total_clinical_cost'] = float(total_cost)

    return metrics


def generate_medical_report(metrics: Dict[str, Any], model_name: str) -> str:
    """
    生成医学评估报告

    Args:
        metrics: 医学指标字典
        model_name: 模型名称

    Returns:
        格式化的医学报告字符串
    """
    report = f"""
=== MEDICAL MODEL EVALUATION REPORT ===
Model: {model_name}
Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== CLINICAL PERFORMANCE SUMMARY ===

CRITICAL PATHOLOGICAL CLASS (Class 2) PERFORMANCE:
- Sensitivity: {metrics['Pathological_sensitivity']:.3f}
  → Ability to detect pathological cases (CRITICAL for patient safety)
- Specificity: {metrics['Pathological_specificity']:.3f}
  → Ability to avoid false alarms
- PPV: {metrics['Pathological_ppv']:.3f}
  → Probability that positive prediction is correct
- NPV: {metrics['Pathological_npv']:.3f}
  → Probability that negative prediction is correct

SUSPECT CLASS (Class 1) PERFORMANCE:
- Sensitivity: {metrics['Suspect_sensitivity']:.3f}
- Specificity: {metrics['Suspect_specificity']:.3f}
- PPV: {metrics['Suspect_ppv']:.3f}
- NPV: {metrics['Suspect_npv']:.3f}

NORMAL CLASS (Class 0) PERFORMANCE:
- Sensitivity: {metrics['Normal_sensitivity']:.3f}
- Specificity: {metrics['Normal_specificity']:.3f}
- PPV: {metrics['Normal_ppv']:.3f}
- NPV: {metrics['Normal_npv']:.3f}

=== OVERALL PERFORMANCE ===
- Overall Accuracy: {metrics['overall_accuracy']:.3f}
- Balanced Accuracy: {metrics['balanced_accuracy']:.3f}

=== COST-SENSITIVE ANALYSIS ===
- Total Clinical Cost: {metrics['total_clinical_cost']:.1f}

=== PATIENT SAFETY ASSESSMENT ===
"""

    pathological_sensitivity = metrics['Pathological_sensitivity']
    if pathological_sensitivity >= 0.95:
        safety_level = "EXCELLENT"
        safety_comment = "Very low risk of missing pathological cases"
    elif pathological_sensitivity >= 0.90:
        safety_level = "GOOD"
        safety_comment = "Acceptable risk of missing pathological cases"
    elif pathological_sensitivity >= 0.80:
        safety_level = "MODERATE"
        safety_comment = "Moderate risk - consider additional screening"
    else:
        safety_level = "POOR"
        safety_comment = "HIGH RISK - Not recommended for clinical use"

    report += f"- Safety Level: {safety_level}\n"
    report += f"- Comment: {safety_comment}\n\n"

    report += "=== CONFUSION MATRIX ===\n"
    report += "         Predicted\n"
    report += "Actual   Normal  Suspect  Pathological\n"
    cm = metrics['confusion_matrix']
    for i in range(3):
        row_name = CLASS_NAMES[i].ljust(12)
        report += f"{row_name}"
        for j in range(3):
            report += f"{cm[i, j]:6d}  "
        report += "\n"

    report += "\n" + "="*50 + "\n"
    return report


def plot_confusion_matrix(cm: np.ndarray, save_path: Path) -> None:
    """绘制混淆矩阵"""
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                xticklabels=[f'{i}\n({CLASS_NAMES[i]})' for i in range(3)],
                yticklabels=[f'{i}\n({CLASS_NAMES[i]})' for i in range(3)])

    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - Cross-Validation Results', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves(y_true: np.ndarray, y_proba: np.ndarray, save_path: Path) -> None:
    """绘制ROC曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, class_label in enumerate([0, 1, 2]):
        ax = axes[i]
        y_true_binary = (y_true == class_label).astype(int)
        y_score = y_proba[:, class_label]

        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        auc_score = roc_auc_score(y_true_binary, y_score)

        ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {CLASS_NAMES[class_label]}\n(Risk: {CLASS_RISK_LEVELS[class_label]})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_shap_analysis(model, X: pd.DataFrame, save_dir: Path, algo: str, max_display: int = 20) -> None:
    """
    生成SHAP可解释性分析（仅支持树模型：RF和XGBoost）

    Args:
        model: 训练好的模型
        X: 特征数据
        save_dir: 保存目录
        algo: 算法名称
        max_display: 最多显示的特征数
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available. Skipping SHAP analysis.")
        return

    # 只对树模型做SHAP分析（快速）
    if algo.lower() not in ['rf', 'xgb']:
        logger.info(f"Skipping SHAP analysis for {algo} (only supported for RF and XGBoost)")
        return

    logger.info("Generating SHAP analysis...")
    save_dir.mkdir(parents=True, exist_ok=True)

    # 提取模型和特征名称
    feature_names = X.columns.tolist()

    if hasattr(model, 'named_steps'):
        base_model = model.named_steps['clf']
        # 通过pipeline转换数据（保留DataFrame以避免警告）
        X_transformed = X.copy()
        for step_name, step in model.steps[:-1]:
            if hasattr(step, 'transform'):
                X_transformed = step.transform(X_transformed)
        # 转为numpy数组用于SHAP
        X_for_shap = np.array(X_transformed)
    else:
        base_model = model
        X_for_shap = X.values

    # 采样（SHAP较慢）
    if len(X_for_shap) > 500:
        logger.info(f"Sampling 500 instances from {len(X_for_shap)} for SHAP analysis")
        sample_indices = np.random.choice(len(X_for_shap), 500, replace=False)
        X_sample = X_for_shap[sample_indices]
    else:
        X_sample = X_for_shap

    logger.info(f"X_sample shape: {X_sample.shape}, type: {type(X_sample)}")

    # 创建SHAP explainer
    try:
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X_sample)
        logger.info("Using TreeExplainer for SHAP analysis")
    except Exception:
        try:
            # 对于非树模型（如SVM），使用KernelExplainer（较慢）
            # 禁用SHAP的详细日志输出
            shap_logger = logging.getLogger('shap')
            original_level = shap_logger.level
            shap_logger.setLevel(logging.WARNING)

            logger.info("Using KernelExplainer (this may take a while for non-tree models)...")
            background = shap.sample(X_for_shap, min(100, len(X_for_shap)))
            explainer = shap.KernelExplainer(base_model.predict_proba, background)
            shap_values = explainer.shap_values(X_sample)

            # 恢复日志级别
            shap_logger.setLevel(original_level)
            logger.info("KernelExplainer analysis completed")
        except Exception as e:
            logger.error(f"SHAP analysis failed: {e}")
            return

    # 处理不同格式的shap_values
    # List格式: [class0_array, class1_array, class2_array] 每个都是 (n_samples, n_features)
    # 3D ndarray格式: (n_samples, n_features, n_classes) - XGBoost常用这种格式

    if isinstance(shap_values, list):
        # List格式 - 每个类别一个数组
        logger.info(f"SHAP values is list with {len(shap_values)} classes")
        shap_list = shap_values
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # 3D array格式 - 转换为list格式
        logger.info(f"SHAP values is 3D array with shape {shap_values.shape}, converting to list format")
        # (n_samples, n_features, n_classes) -> list of (n_samples, n_features)
        shap_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    else:
        logger.warning(f"Unexpected SHAP values format: {type(shap_values)}, shape: {shap_values.shape if hasattr(shap_values, 'shape') else 'N/A'}")
        return

    # 绘制全局特征重要性（所有类别平均）
    shap_abs_mean = np.mean([np.abs(sv).mean(axis=0) for sv in shap_list], axis=0)
    sorted_idx = np.argsort(shap_abs_mean)[-max_display:]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_idx)), shap_abs_mean[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Mean |SHAP value| (averaged across all classes)')
    plt.title('Global Feature Importance (SHAP)')
    plt.tight_layout()
    plt.savefig(save_dir / 'shap_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved SHAP feature importance plot")

    # 每个类别的SHAP summary
    for class_idx, class_name in CLASS_NAMES.items():
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_list[class_idx], X_sample,
                        feature_names=feature_names,
                        max_display=max_display, show=False)
        plt.title(f'SHAP Summary - {class_name} Class')
        plt.tight_layout()
        plt.savefig(save_dir / f'shap_summary_class_{class_idx}_{class_name.lower()}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved SHAP summary plot for {class_name} class")

    logger.info(f"SHAP analysis completed and saved to {save_dir}")


def train_main():
    """
    主训练流程

    解析命令行参数、加载数据、训练模型、评估性能并保存结果
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="CTG Training Pipeline")
    parser.add_argument("--algo", type=str, default="rf", choices=["logreg", "rf", "xgb", "lgbm", "svm"],
                        help="Classification algorithm to use")
    parser.add_argument("--use-smote", action="store_true", help="Enable SMOTE oversampling")
    parser.add_argument("--class-weight", type=str, default=None, choices=[None, "balanced", "balanced_subsample"],
                        help="Class weight strategy")
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds (stratified)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--doctor-col", type=str, default=None,
                        help="Column name for obstetrician ID for per-doctor validation")
    parser.add_argument("--time-col", type=str, default=None,
                        help="Column name for timestamp to use time-series split")
    parser.add_argument("--target", type=str, default="NSP", help="Target column name")
    parser.add_argument("--use-normalized", action="store_true",
                        help="Use normalized data instead of clean data (default: False)")
    parser.add_argument("--mlflow-uri", type=str, default=None, help="MLflow tracking URI")
    parser.add_argument("--mlflow-exp", type=str, default="CTG-Experiments", help="MLflow experiment name")
    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # 生成时间戳，用于文件夹命名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 设置 MLflow tracking URI 为 models/mlruns
    mlflow_tracking_uri = args.mlflow_uri or "file:./models/mlruns"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_exp)

    # 加载数据（根据命令行参数选择标准化或原始数据）
    df = load_data(use_normalized=args.use_normalized)
    X, y = get_features_and_target(df, target=args.target)

    # 将标签从 [1,2,3] 映射到 [0,1,2]
    y = y - 1
    logger.info(f"Target classes after remapping: {np.unique(y)}")

    # 提取分组信息（如果提供）
    groups = None
    if args.doctor_col and args.doctor_col in df.columns:
        groups = df.loc[y.index, args.doctor_col]

    # 使用时间戳作为文件夹名称
    model_out_dir = Path("models") / "saved" / f"{args.algo}_{timestamp}"
    model_out_path = model_out_dir / "model.joblib"

    # 开始 MLflow run
    with mlflow.start_run(run_name=f"{args.algo}-{timestamp}"):
        # 记录训练参数
        params = {
            "algo": args.algo,
            "use_smote": args.use_smote,
            "class_weight": args.class_weight,
            "cv": args.cv,
            "random_state": args.random_state,
            "target": args.target,
            "use_normalized": args.use_normalized,
        }
        mlflow.log_params(params)

        # 执行交叉验证
        logger.info("Starting cross-validation...")
        time_split = args.time_col is not None and args.time_col in df.columns
        scores = run_cv(
            X, y, args.algo, args.use_smote, args.class_weight,
            args.cv, args.random_state, groups=groups, time_series_split=time_split
        )

        # 在全部数据上训练最终模型
        logger.info("Training final model on full dataset...")
        clf = build_classifier(args.algo, class_weight=args.class_weight, random_state=args.random_state)
        steps: List[Tuple[str, object]] = [("scaler", StandardScaler(with_mean=False))]
        if args.use_smote:
            steps.append(("smote", SMOTE(random_state=args.random_state)))
        steps.append(("clf", clf))
        model = ImbPipeline(steps=steps)

        # 为XGBoost使用sample_weight
        if args.algo.lower() == 'xgb' and args.class_weight:
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight('balanced', y)
            model.fit(X, y, clf__sample_weight=sample_weights)
        else:
            model.fit(X, y)

        # 可选的按医生验证
        if groups is not None:
            logger.info("Performing per-doctor validation...")
            pd_scores = per_doctor_validation(X, y, groups, args.algo, args.use_smote, args.class_weight, args.random_state)
            scores.update(pd_scores)

        # 提取交叉验证的预测结果
        cv_predictions = scores.pop('cv_predictions', None)

        # 保存模型和产物
        logger.info(f"Saving model to {model_out_path}")
        log_model_artifacts("model", model, args.algo, list(X.columns), scores, model_out_path, params, timestamp)

        # ===== 医学评估和SHAP分析 =====
        if cv_predictions is not None:
            logger.info("=" * 60)
            logger.info("Starting detailed medical evaluation...")

            # 设置评估结果输出目录
            eval_output_dir = Path("reports") / "new" / "evaluation" / f"{args.algo}_{timestamp}"
            eval_output_dir.mkdir(parents=True, exist_ok=True)

            y_true_cv = cv_predictions['y_true']
            y_pred_cv = cv_predictions['y_pred']
            y_proba_cv = cv_predictions['y_proba']

            # 计算医学指标
            logger.info("Calculating medical metrics...")
            medical_metrics = calculate_medical_metrics(y_true_cv, y_pred_cv, y_proba_cv)

            # 生成医学报告
            model_name = f"{args.algo.upper()}_{timestamp}_CV"
            medical_report = generate_medical_report(medical_metrics, model_name)

            # 保存医学报告
            report_path = eval_output_dir / "medical_evaluation_report.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(medical_report)
            logger.info(f"Medical report saved to {report_path}")

            # 保存医学指标为JSON
            # 移除numpy数组以便JSON序列化
            metrics_for_json = {k: v for k, v in medical_metrics.items() if k != 'confusion_matrix'}
            metrics_for_json['confusion_matrix'] = medical_metrics['confusion_matrix'].tolist()

            metrics_path = eval_output_dir / "medical_metrics.json"
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_for_json, f, indent=2, ensure_ascii=False)
            logger.info(f"Medical metrics saved to {metrics_path}")

            # 绘制混淆矩阵
            logger.info("Plotting confusion matrix...")
            cm_path = eval_output_dir / "confusion_matrix_cv.png"
            plot_confusion_matrix(medical_metrics['confusion_matrix'], cm_path)

            # 绘制ROC曲线
            if y_proba_cv is not None:
                logger.info("Plotting ROC curves...")
                roc_path = eval_output_dir / "roc_curves_cv.png"
                plot_roc_curves(y_true_cv, y_proba_cv, roc_path)

            # SHAP分析（使用最终模型，仅RF和XGBoost）
            logger.info("Generating SHAP analysis for model interpretability...")
            shap_dir = eval_output_dir / "shap"
            generate_shap_analysis(model, X, shap_dir, args.algo, max_display=20)

            # 打印医学报告
            print(medical_report)

            logger.info(f"All evaluation results saved to: {eval_output_dir}")
            logger.info("=" * 60)

        logger.info("=" * 60)
        logger.info(f"Training completed successfully!")
        logger.info(f"Model saved to: {model_out_dir}")
        logger.info(f"Evaluation results saved to: {eval_output_dir if cv_predictions else 'N/A'}")
        logger.info(f"CV Results: {scores}")
        logger.info("=" * 60)


if __name__ == "__main__":
    train_main()