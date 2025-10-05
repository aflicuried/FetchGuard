"""
Vertical Comparison of XGBoost Models
Compare XGBoost configurations with different data preprocessing and balancing strategies
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# Define XGBoost models with different configurations
XGB_MODELS = {
    'XGB-Baseline': {
        'summary': 'models/saved/xgb_20251005_171512/training_summary.json',
        'medical': 'reports/new/evaluation/xgb_20251005_171512/medical_metrics.json',
        'config': 'Clean data, no SMOTE, no class_weight'
    },
    'XGB-Normalized': {
        'summary': 'models/saved/xgb_20251005_172418/training_summary.json',
        'medical': 'reports/new/evaluation/xgb_20251005_172418/medical_metrics.json',
        'config': 'Normalized data, no SMOTE, no class_weight'
    },
    'XGB-Balanced': {
        'summary': 'models/saved/xgb_20251005_180634/training_summary.json',
        'medical': 'reports/new/evaluation/xgb_20251005_180634/medical_metrics.json',
        'config': 'Clean data, no SMOTE, sample_weight=balanced'
    }
}

def load_xgb_data():
    """Load all XGBoost model data"""
    data = {}

    for model_name, paths in XGB_MODELS.items():
        with open(paths['summary'], 'r') as f:
            summary = json.load(f)

        with open(paths['medical'], 'r') as f:
            medical = json.load(f)

        data[model_name] = {
            'config': paths['config'],
            'use_smote': summary['parameters']['use_smote'],
            'class_weight': summary['parameters']['class_weight'],
            'use_normalized': summary['parameters']['use_normalized'],
            'accuracy': summary['metrics']['acc'],
            'balanced_accuracy': summary['metrics']['bacc'],
            'f1_macro': summary['metrics']['f1_macro'],
            'roc_auc': summary['metrics']['roc_auc_ovr'],
            'pathological_sensitivity': medical['Pathological_sensitivity'],
            'pathological_specificity': medical['Pathological_specificity'],
            'pathological_ppv': medical['Pathological_ppv'],
            'pathological_npv': medical['Pathological_npv'],
            'suspect_sensitivity': medical['Suspect_sensitivity'],
            'suspect_specificity': medical['Suspect_specificity'],
            'normal_sensitivity': medical['Normal_sensitivity'],
            'normal_specificity': medical['Normal_specificity'],
            'clinical_cost': medical['total_clinical_cost'],
            'confusion_matrix': np.array(medical['confusion_matrix'])
        }

    return data

def plot_config_comparison(data, save_path):
    """Plot configuration impact on performance"""
    models = list(data.keys())

    metrics = ['accuracy', 'balanced_accuracy', 'f1_macro', 'roc_auc']
    metric_labels = ['Accuracy', 'Balanced\nAccuracy', 'F1 Macro', 'ROC AUC']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    colors = ['#3498db', '#2ecc71', '#e74c3c']

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        values = [data[m][metric] for m in models]

        bars = ax.bar(range(len(models)), values, color=colors, alpha=0.8,
                     edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)

        ax.set_ylabel(label, fontweight='bold', fontsize=11)
        ax.set_title(f'{label} across XGBoost Configurations', fontweight='bold', fontsize=12)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.set_ylim([min(values) - 0.02, 1.0])
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    plt.suptitle('XGBoost Performance Metrics Comparison',
                fontweight='bold', fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_class_sensitivity(data, save_path):
    """Plot sensitivity for all classes across XGBoost models"""
    models = list(data.keys())
    classes = ['Normal', 'Suspect', 'Pathological']

    sensitivity_data = {
        'Normal': [data[m]['normal_sensitivity'] for m in models],
        'Suspect': [data[m]['suspect_sensitivity'] for m in models],
        'Pathological': [data[m]['pathological_sensitivity'] for m in models]
    }

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(models))
    width = 0.25
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    for idx, class_name in enumerate(classes):
        offset = (idx - 1) * width
        bars = ax.bar(x + offset, sensitivity_data[class_name], width,
                     label=class_name, color=colors[idx], alpha=0.8,
                     edgecolor='black', linewidth=1.2)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('XGBoost Configurations', fontweight='bold', fontsize=12)
    ax.set_ylabel('Sensitivity (Recall)', fontweight='bold', fontsize=12)
    ax.set_title('Class-Specific Sensitivity across XGBoost Configurations',
                fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.legend(title='Class', fontsize=11, title_fontsize=12, loc='lower right')
    ax.set_ylim([0.7, 1.05])
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Add safety threshold
    ax.axhline(y=0.90, color='red', linestyle='--', alpha=0.4,
              label='Safety Threshold (0.90)')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_pathological_metrics(data, save_path):
    """Plot detailed pathological class metrics"""
    models = list(data.keys())
    metrics = ['pathological_sensitivity', 'pathological_specificity',
               'pathological_ppv', 'pathological_npv']
    metric_labels = ['Sensitivity', 'Specificity', 'PPV', 'NPV']

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(models))
    width = 0.2
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        offset = (idx - 1.5) * width
        values = [data[m][metric] for m in models]

        bars = ax.bar(x + offset, values, width, label=label,
                     color=colors[idx], alpha=0.8,
                     edgecolor='black', linewidth=1.0)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('XGBoost Configurations', fontweight='bold', fontsize=12)
    ax.set_ylabel('Metric Value', fontweight='bold', fontsize=12)
    ax.set_title('Pathological Class Detailed Metrics (Critical for Patient Safety)',
                fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.legend(fontsize=11, ncol=4, loc='lower right')
    ax.set_ylim([0.8, 1.02])
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_clinical_cost_comparison(data, save_path):
    """Plot clinical cost comparison"""
    models = list(data.keys())
    costs = [data[m]['clinical_cost'] for m in models]
    configs = [data[m]['config'] for m in models]

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars = ax.bar(models, costs, color=colors, alpha=0.8,
                 edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax.set_ylabel('Total Clinical Cost', fontweight='bold', fontsize=12)
    ax.set_title('Clinical Cost Comparison across XGBoost Configurations\n(Lower is Better)',
                fontweight='bold', fontsize=14)
    ax.tick_params(axis='x', rotation=15)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Add reference line
    ax.axhline(y=300, color='green', linestyle='--', alpha=0.5,
              label='Excellent Performance (<300)', linewidth=2)
    ax.legend(fontsize=11)

    # Add config text below x-axis
    for idx, (bar, config) in enumerate(zip(bars, configs)):
        ax.text(bar.get_x() + bar.get_width()/2., -20,
               config, ha='center', va='top', fontsize=8,
               style='italic')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_confusion_matrices(data, save_path):
    """Plot confusion matrices for all XGBoost models"""
    models = list(data.keys())
    class_names = ['Normal', 'Suspect', 'Pathological']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, model_name in enumerate(models):
        ax = axes[idx]
        cm = data[model_name]['confusion_matrix']
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues', ax=ax,
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Normalized Count'},
                   vmin=0, vmax=1)

        ax.set_xlabel('Predicted Class', fontweight='bold')
        ax.set_ylabel('True Class', fontweight='bold')
        ax.set_title(f'{model_name}\n{data[model_name]["config"]}',
                    fontweight='bold', fontsize=10)

    plt.suptitle('Confusion Matrices Comparison - XGBoost Models',
                fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_performance_radar(data, save_path):
    """Create radar chart comparing XGBoost configurations"""
    from math import pi

    models = list(data.keys())
    categories = ['Accuracy', 'Bal. Acc', 'F1 Macro',
                 'Path. Sens', 'Path. Spec', 'ROC AUC']

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Number of variables
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    colors = ['#3498db', '#2ecc71', '#e74c3c']

    for idx, model_name in enumerate(models):
        d = data[model_name]
        values = [
            d['accuracy'],
            d['balanced_accuracy'],
            d['f1_macro'],
            d['pathological_sensitivity'],
            d['pathological_specificity'],
            d['roc_auc']
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=model_name,
               color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0.8, 1.0)
    ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_yticklabels(['0.80', '0.85', '0.90', '0.95', '1.00'])
    ax.grid(True)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.title('XGBoost Performance Radar Chart', fontweight='bold',
             fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def create_xgb_summary_table(data, save_path):
    """Create comprehensive XGBoost comparison table"""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('tight')
    ax.axis('off')

    models = list(data.keys())
    headers = ['Model', 'Configuration', 'Accuracy', 'Bal. Acc', 'F1 Macro',
               'Path. Sens', 'Path. Spec', 'Cost']

    table_data = []
    for model in models:
        d = data[model]
        row = [
            model,
            d['config'],
            f"{d['accuracy']:.4f}",
            f"{d['balanced_accuracy']:.4f}",
            f"{d['f1_macro']:.4f}",
            f"{d['pathological_sensitivity']:.4f}",
            f"{d['pathological_specificity']:.4f}",
            f"{int(d['clinical_cost'])}"
        ]
        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='left', loc='center',
                    colWidths=[0.13, 0.32, 0.10, 0.10, 0.10, 0.10, 0.10, 0.08])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 3)

    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')

    # Style data rows
    colors = ['#ecf0f1', '#ffffff']
    for i in range(1, len(models) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            cell.set_facecolor(colors[i % 2])

            # Highlight best values
            if j == 2:  # Accuracy
                if float(table_data[i-1][j]) >= max([float(data[m]['accuracy']) for m in models]) - 0.0001:
                    cell.set_facecolor('#2ecc71')
                    cell.set_text_props(weight='bold')
            elif j == 5:  # Path. Sens
                if float(table_data[i-1][j]) == max([float(data[m]['pathological_sensitivity']) for m in models]):
                    cell.set_facecolor('#2ecc71')
                    cell.set_text_props(weight='bold')
            elif j == 7:  # Cost (lower is better)
                if float(table_data[i-1][j]) == min([float(data[m]['clinical_cost']) for m in models]):
                    cell.set_facecolor('#2ecc71')
                    cell.set_text_props(weight='bold')

    plt.title('XGBoost Configuration Comparison Summary',
             fontweight='bold', fontsize=16, pad=20)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def main():
    """Main execution"""
    print("="*60)
    print("XGBOOST MODEL COMPARISON")
    print("="*60)

    # Create output directory
    output_dir = Path("plot/xgb_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading XGBoost model data...")
    data = load_xgb_data()

    # Generate plots
    print("\nGenerating XGBoost comparison plots...")
    plot_config_comparison(data, output_dir / "01_performance_metrics.png")
    plot_class_sensitivity(data, output_dir / "02_class_sensitivity.png")
    plot_pathological_metrics(data, output_dir / "03_pathological_metrics.png")
    plot_clinical_cost_comparison(data, output_dir / "04_clinical_cost.png")
    plot_confusion_matrices(data, output_dir / "05_confusion_matrices.png")
    plot_performance_radar(data, output_dir / "06_performance_radar.png")
    create_xgb_summary_table(data, output_dir / "07_summary_table.png")

    print("\n" + "="*60)
    print("XGBOOST MODEL COMPARISON COMPLETED!")
    print(f"All plots saved to: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
