"""
CTG 神经网络分类训练脚本

使用多层感知机(MLP)进行CTG胎儿心电图分类
包含完整的正则化策略和医学评估指标
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

import mlflow
import mlflow.pytorch

logger = logging.getLogger(__name__)

# Medical class definitions
CLASS_NAMES = {0: "Normal", 1: "Suspect", 2: "Pathological"}
CLASS_RISK_LEVELS = {0: "Low", 1: "Moderate", 2: "High"}

# Cost matrix for medical decision making
DEFAULT_COST_MATRIX = np.array([
    [0,   1,   5],
    [1,   0,   3],
    [20,  10,  0]
])


class MLPClassifier(nn.Module):
    """
    多层感知机分类器，包含正则化策略

    架构:
    - 输入层 -> 隐藏层1 -> Dropout -> 隐藏层2 -> Dropout -> 输出层
    - 使用BatchNorm稳定训练
    - 使用Dropout防止过拟合
    - 使用ReLU激活函数
    """
    def __init__(self, input_dim: int, hidden_dims: list = [128, 64],
                 num_classes: int = 3, dropout: float = 0.3):
        super(MLPClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_data(use_normalized: bool = False) -> pd.DataFrame:
    """加载CTG数据集"""
    if use_normalized:
        data_path = Path("data/processed/ctg_normalized.csv")
    else:
        data_path = Path("data/processed/ctg_clean.csv")

    if not data_path.exists():
        raise FileNotFoundError(f"Data not found at {data_path}")

    logger.info(f"Loading data from {data_path}")
    return pd.read_csv(data_path)


def get_features_and_target(df: pd.DataFrame, target: str = "NSP") -> Tuple[pd.DataFrame, pd.Series]:
    """从数据集中分离特征和目标变量"""
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found")

    y = pd.to_numeric(df[target], errors="coerce")
    mask = y.notna()
    y = y.loc[mask].astype(int)

    exclude_cols = [target, 'CLASS']
    X = df.loc[mask].select_dtypes(include=[np.number]).drop(
        columns=[col for col in exclude_cols if col in df],
        errors="ignore"
    )

    logger.info(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    return X, y


def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    return total_loss / len(loader), correct / total


def evaluate_epoch(model, loader, criterion, device):
    """评估一个epoch"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return total_loss / len(loader), np.array(all_preds), np.array(all_labels), np.array(all_probs)


def calculate_medical_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """计算医学评估指标"""
    metrics = {}

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    metrics['confusion_matrix'] = cm

    # 每个类别的指标
    for class_label in [0, 1, 2]:
        y_true_binary = (y_true == class_label).astype(int)
        y_pred_binary = (y_pred == class_label).astype(int)

        cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
        if cm_binary.shape == (2, 2):
            tn, fp, fn, tp = cm_binary.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

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

    # 临床成本
    total_cost = np.sum(cm * DEFAULT_COST_MATRIX)
    metrics['total_clinical_cost'] = float(total_cost)

    return metrics


def generate_medical_report(metrics: Dict, model_name: str) -> str:
    """生成医学评估报告"""
    report = f"""
=== NEURAL NETWORK MEDICAL EVALUATION REPORT ===
Model: {model_name}
Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== CLINICAL PERFORMANCE SUMMARY ===

CRITICAL PATHOLOGICAL CLASS (Class 2) PERFORMANCE:
- Sensitivity: {metrics['Pathological_sensitivity']:.3f}
  → Ability to detect pathological cases (CRITICAL for patient safety)
- Specificity: {metrics['Pathological_specificity']:.3f}
- PPV: {metrics['Pathological_ppv']:.3f}
- NPV: {metrics['Pathological_npv']:.3f}

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


def plot_confusion_matrix(cm: np.ndarray, save_path: Path):
    """绘制混淆矩阵"""
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                xticklabels=[f'{i}\n({CLASS_NAMES[i]})' for i in range(3)],
                yticklabels=[f'{i}\n({CLASS_NAMES[i]})' for i in range(3)])

    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - Neural Network CV Results', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves(y_true: np.ndarray, y_proba: np.ndarray, save_path: Path):
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


def train_nn():
    """神经网络训练主函数"""
    parser = argparse.ArgumentParser(description="CTG Neural Network Training")
    parser.add_argument("--hidden-dims", type=int, nargs='+', default=[128, 64],
                        help="Hidden layer dimensions")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="L2 regularization")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--cv", type=int, default=5, help="CV folds")
    parser.add_argument("--use-normalized", action="store_true", help="Use normalized data")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # 设置随机种子
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # MLflow设置
    mlflow.set_tracking_uri("file:./models/mlruns")
    mlflow.set_experiment("CTG-NeuralNetwork")

    # 加载数据
    df = load_data(use_normalized=args.use_normalized)
    X, y = get_features_and_target(df)

    # 标签映射 [1,2,3] -> [0,1,2]
    y = y - 1

    # 交叉验证
    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)

    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    fold_metrics = []

    with mlflow.start_run(run_name=f"mlp-{timestamp}"):
        # 记录参数
        mlflow.log_params({
            "hidden_dims": args.hidden_dims,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "cv": args.cv,
            "use_normalized": args.use_normalized
        })

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            logger.info(f"Training Fold {fold}/{args.cv}")

            # 分割数据
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 转换为Tensor
            X_train_t = torch.FloatTensor(X_train_scaled)
            y_train_t = torch.LongTensor(y_train.values)
            X_test_t = torch.FloatTensor(X_test_scaled)
            y_test_t = torch.LongTensor(y_test.values)

            # DataLoader
            train_dataset = TensorDataset(X_train_t, y_train_t)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            test_dataset = TensorDataset(X_test_t, y_test_t)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

            # 模型
            model = MLPClassifier(
                input_dim=X_train.shape[1],
                hidden_dims=args.hidden_dims,
                num_classes=3,
                dropout=args.dropout
            ).to(device)

            # 优化器和损失函数（包含L2正则化）
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            # 训练
            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(args.epochs):
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_preds, val_labels, val_probs = evaluate_epoch(model, test_loader, criterion, device)

                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break

            # 最终评估
            _, fold_preds, fold_labels, fold_probs = evaluate_epoch(model, test_loader, criterion, device)

            fold_acc = accuracy_score(fold_labels, fold_preds)
            fold_bacc = balanced_accuracy_score(fold_labels, fold_preds)
            logger.info(f"Fold {fold}: Accuracy={fold_acc:.3f}, Balanced Accuracy={fold_bacc:.3f}")

            # 收集结果
            all_y_true.extend(fold_labels)
            all_y_pred.extend(fold_preds)
            all_y_proba.extend(fold_probs)
            fold_metrics.append({'acc': fold_acc, 'bacc': fold_bacc})

        # 整体评估
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        all_y_proba = np.array(all_y_proba)

        # 计算医学指标
        medical_metrics = calculate_medical_metrics(all_y_true, all_y_pred)

        # 记录MLflow指标
        avg_acc = np.mean([m['acc'] for m in fold_metrics])
        avg_bacc = np.mean([m['bacc'] for m in fold_metrics])
        mlflow.log_metric("cv_accuracy", avg_acc)
        mlflow.log_metric("cv_balanced_accuracy", avg_bacc)

        # 生成报告
        model_name = f"MLP_{timestamp}_CV"
        medical_report = generate_medical_report(medical_metrics, model_name)

        # 保存结果
        eval_output_dir = Path("reports") / "new" / "evaluation" / f"mlp_{timestamp}"
        eval_output_dir.mkdir(parents=True, exist_ok=True)

        # 保存报告
        report_path = eval_output_dir / "medical_evaluation_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(medical_report)

        # 保存指标
        metrics_for_json = {k: v for k, v in medical_metrics.items() if k != 'confusion_matrix'}
        metrics_for_json['confusion_matrix'] = medical_metrics['confusion_matrix'].tolist()

        metrics_path = eval_output_dir / "medical_metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_for_json, f, indent=2, ensure_ascii=False)

        # 绘图
        cm_path = eval_output_dir / "confusion_matrix_cv.png"
        plot_confusion_matrix(medical_metrics['confusion_matrix'], cm_path)

        roc_path = eval_output_dir / "roc_curves_cv.png"
        plot_roc_curves(all_y_true, all_y_proba, roc_path)

        # 训练最终模型（使用全部数据）
        logger.info("Training final model on full dataset...")
        scaler_final = StandardScaler()
        X_scaled_final = scaler_final.fit_transform(X)

        X_tensor = torch.FloatTensor(X_scaled_final)
        y_tensor = torch.LongTensor(y.values)

        dataset_final = TensorDataset(X_tensor, y_tensor)
        loader_final = DataLoader(dataset_final, batch_size=args.batch_size, shuffle=True)

        final_model = MLPClassifier(
            input_dim=X.shape[1],
            hidden_dims=args.hidden_dims,
            num_classes=3,
            dropout=args.dropout
        ).to(device)

        optimizer_final = optim.Adam(final_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(args.epochs):
            train_loss, _ = train_epoch(final_model, loader_final, criterion, optimizer_final, device)
            if (epoch + 1) % 20 == 0:
                logger.info(f"Final Model Epoch {epoch+1}: Loss={train_loss:.4f}")

        # 保存模型
        model_dir = Path("models") / "saved" / f"mlp_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': final_model.state_dict(),
            'scaler': scaler_final,
            'input_dim': X.shape[1],
            'hidden_dims': args.hidden_dims,
            'dropout': args.dropout
        }, model_dir / "model.pth")

        # 保存训练摘要
        training_summary = {
            # 模型配置
            'model_type': 'MLP',
            'architecture': {
                'input_dim': X.shape[1],
                'hidden_dims': args.hidden_dims,
                'output_dim': 3,
                'dropout': args.dropout
            },

            # 训练超参数
            'hyperparameters': {
                'learning_rate': args.lr,
                'weight_decay': args.weight_decay,
                'batch_size': args.batch_size,
                'max_epochs': args.epochs,
                'early_stopping_patience': args.patience
            },

            # 数据配置
            'data': {
                'use_normalized': args.use_normalized,
                'data_path': 'data/processed/ctg_normalized.csv' if args.use_normalized else 'data/processed/ctg_clean.csv',
                'num_samples': len(X),
                'num_features': X.shape[1],
                'feature_names': X.columns.tolist()
            },

            # 交叉验证配置
            'cross_validation': {
                'method': 'StratifiedKFold',
                'n_splits': args.cv,
                'random_state': args.random_state
            },

            # 模型性能
            'performance': {
                'cv_accuracy': float(avg_acc),
                'cv_balanced_accuracy': float(avg_bacc),
                'fold_results': [
                    {
                        'fold': i+1,
                        'accuracy': float(m['acc']),
                        'balanced_accuracy': float(m['bacc'])
                    }
                    for i, m in enumerate(fold_metrics)
                ]
            },

            # 医学指标
            'medical_metrics': {
                'pathological_sensitivity': float(medical_metrics['Pathological_sensitivity']),
                'pathological_specificity': float(medical_metrics['Pathological_specificity']),
                'pathological_ppv': float(medical_metrics['Pathological_ppv']),
                'pathological_npv': float(medical_metrics['Pathological_npv']),
                'suspect_sensitivity': float(medical_metrics['Suspect_sensitivity']),
                'suspect_specificity': float(medical_metrics['Suspect_specificity']),
                'normal_sensitivity': float(medical_metrics['Normal_sensitivity']),
                'normal_specificity': float(medical_metrics['Normal_specificity']),
                'overall_accuracy': float(medical_metrics['overall_accuracy']),
                'balanced_accuracy': float(medical_metrics['balanced_accuracy']),
                'total_clinical_cost': float(medical_metrics['total_clinical_cost'])
            },

            # 元数据
            'metadata': {
                'timestamp': timestamp,
                'device': str(device),
                'framework': 'PyTorch',
                'model_path': str(model_dir / "model.pth"),
                'evaluation_path': str(eval_output_dir)
            }
        }

        summary_path = model_dir / "training_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(training_summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Model saved to {model_dir}")
        logger.info(f"Training summary saved to {summary_path}")
        logger.info(f"Evaluation results saved to {eval_output_dir}")

        print(medical_report)

        logger.info("=" * 60)
        logger.info("Neural Network Training Completed!")
        logger.info("=" * 60)


if __name__ == "__main__":
    train_nn()