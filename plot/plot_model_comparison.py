"""
Horizontal Comparison of Baseline Models
Compare LogReg, RF, XGBoost, SVM, and Neural Network without special parameters
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

# Define baseline models (no SMOTE, no class_weight)
MODELS = {
    'LogReg': {
        'summary': 'models/saved/logreg_20251005_175322/training_summary.json',
        'medical': 'reports/new/evaluation/logreg_20251005_175322/medical_metrics.json'
    },
    'Random Forest': {
        'summary': 'models/saved/rf_20251005_171629/training_summary.json',
        'medical': 'reports/new/evaluation/rf_20251005_171629/medical_metrics.json'
    },
    'XGBoost': {
        'summary': 'models/saved/xgb_20251005_171512/training_summary.json',
        'medical': 'reports/new/evaluation/xgb_20251005_171512/medical_metrics.json'
    },
    'SVM': {
        'summary': 'models/saved/svm_20251005_175237/training_summary.json',
        'medical': 'reports/new/evaluation/svm_20251005_175237/medical_metrics.json'
    },
    'Neural Network': {
        'summary': 'models/saved/mlp_20251005_183641/training_summary.json',
        'medical': 'reports/new/evaluation/mlp_20251005_183641/medical_metrics.json'
    }
}

def load_data():
    """Load all model data"""
    data = {}

    for model_name, paths in MODELS.items():
        with open(paths['summary'], 'r') as f:
            summary = json.load(f)

        with open(paths['medical'], 'r') as f:
            medical = json.load(f)

        # Extract metrics
        if model_name == 'Neural Network':
            acc = summary['performance']['cv_accuracy']
            bacc = summary['performance']['cv_balanced_accuracy']
            roc_auc = None  # Not in NN summary
        else:
            acc = summary['metrics']['acc']
            bacc = summary['metrics']['bacc']
            roc_auc = summary['metrics']['roc_auc_ovr']

        data[model_name] = {
            'accuracy': acc,
            'balanced_accuracy': bacc,
            'roc_auc': roc_auc,
            'pathological_sensitivity': medical['Pathological_sensitivity'],
            'pathological_specificity': medical['Pathological_specificity'],
            'suspect_sensitivity': medical['Suspect_sensitivity'],
            'normal_sensitivity': medical['Normal_sensitivity'],
            'clinical_cost': medical['total_clinical_cost']
        }

    return data

def plot_performance_metrics(data, save_path):
    """Plot overall performance metrics"""
    models = list(data.keys())
    metrics = ['accuracy', 'balanced_accuracy', 'roc_auc']
    metric_labels = ['Accuracy', 'Balanced Accuracy', 'ROC AUC']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        values = [data[m][metric] if data[m][metric] is not None else 0 for m in models]
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

        bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontweight='bold', fontsize=9)

        ax.set_ylabel(label, fontweight='bold')
        ax.set_title(f'{label} Comparison', fontweight='bold', fontsize=12)
        ax.set_ylim([0, 1.0])
        ax.tick_params(axis='x', rotation=15)

        # Add grid
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_sensitivity_comparison(data, save_path):
    """Plot sensitivity for all classes"""
    models = list(data.keys())
    classes = ['Normal', 'Suspect', 'Pathological']

    # Prepare data
    sensitivity_data = {
        'Normal': [data[m]['normal_sensitivity'] for m in models],
        'Suspect': [data[m]['suspect_sensitivity'] for m in models],
        'Pathological': [data[m]['pathological_sensitivity'] for m in models]
    }

    fig, ax = plt.subplots(figsize=(12, 6))

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
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Models', fontweight='bold', fontsize=12)
    ax.set_ylabel('Sensitivity (Recall)', fontweight='bold', fontsize=12)
    ax.set_title('Class-Specific Sensitivity Comparison', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.legend(title='Class', fontsize=10, title_fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_clinical_cost(data, save_path):
    """Plot clinical cost comparison"""
    models = list(data.keys())
    costs = [data[m]['clinical_cost'] for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    bars = ax.bar(models, costs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_ylabel('Total Clinical Cost', fontweight='bold', fontsize=12)
    ax.set_title('Clinical Cost Comparison (Lower is Better)', fontweight='bold', fontsize=14)
    ax.tick_params(axis='x', rotation=15)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Add reference line for good performance
    ax.axhline(y=500, color='red', linestyle='--', alpha=0.5, label='Target Threshold (500)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_pathological_performance(data, save_path):
    """Plot pathological class performance (most critical)"""
    models = list(data.keys())
    sensitivity = [data[m]['pathological_sensitivity'] for m in models]
    specificity = [data[m]['pathological_specificity'] for m in models]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, sensitivity, width, label='Sensitivity',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, specificity, width, label='Specificity',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Models', fontweight='bold', fontsize=12)
    ax.set_ylabel('Performance', fontweight='bold', fontsize=12)
    ax.set_title('Pathological Class Performance (Most Critical for Patient Safety)',
                fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.05])

    # Add safety threshold lines
    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.4, label='Excellent (≥0.95)')
    ax.axhline(y=0.90, color='orange', linestyle='--', alpha=0.4, label='Good (≥0.90)')

    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def create_summary_table(data, save_path):
    """Create a comprehensive summary table"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    models = list(data.keys())
    headers = ['Model', 'Accuracy', 'Bal. Acc', 'ROC AUC',
               'Path. Sens', 'Path. Spec', 'Clinical Cost']

    table_data = []
    for model in models:
        d = data[model]
        row = [
            model,
            f"{d['accuracy']:.4f}",
            f"{d['balanced_accuracy']:.4f}",
            f"{d['roc_auc']:.4f}" if d['roc_auc'] else 'N/A',
            f"{d['pathological_sensitivity']:.4f}",
            f"{d['pathological_specificity']:.4f}",
            f"{int(d['clinical_cost'])}"
        ]
        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.18, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')

    # Style data rows with alternating colors
    colors = ['#ecf0f1', '#ffffff']
    for i in range(1, len(models) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            cell.set_facecolor(colors[i % 2])

            # Highlight best values
            if j == 1:  # Accuracy
                if float(table_data[i-1][j]) == max([float(data[m]['accuracy']) for m in models]):
                    cell.set_facecolor('#2ecc71')
                    cell.set_text_props(weight='bold', color='white')
            elif j == 4:  # Path. Sens
                if float(table_data[i-1][j]) == max([float(data[m]['pathological_sensitivity']) for m in models]):
                    cell.set_facecolor('#2ecc71')
                    cell.set_text_props(weight='bold', color='white')
            elif j == 6:  # Cost (lower is better)
                if float(table_data[i-1][j]) == min([float(data[m]['clinical_cost']) for m in models]):
                    cell.set_facecolor('#2ecc71')
                    cell.set_text_props(weight='bold', color='white')

    plt.title('Model Performance Summary Table', fontweight='bold', fontsize=16, pad=20)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def main():
    """Main execution"""
    print("="*60)
    print("BASELINE MODEL COMPARISON")
    print("="*60)

    # Create output directory
    output_dir = Path("plot/model_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading model data...")
    data = load_data()

    # Generate plots
    print("\nGenerating comparison plots...")
    plot_performance_metrics(data, output_dir / "01_performance_metrics.png")
    plot_sensitivity_comparison(data, output_dir / "02_sensitivity_comparison.png")
    plot_pathological_performance(data, output_dir / "03_pathological_performance.png")
    plot_clinical_cost(data, output_dir / "04_clinical_cost.png")
    create_summary_table(data, output_dir / "05_summary_table.png")

    print("\n" + "="*60)
    print("MODEL COMPARISON COMPLETED!")
    print(f"All plots saved to: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
