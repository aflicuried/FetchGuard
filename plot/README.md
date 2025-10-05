# CTG Model Visualization Scripts

This directory contains Python scripts to generate all presentation-ready visualizations for the FetalGuard CTG classification project.

## Overview

Four main plotting scripts generate comprehensive visualizations comparing different models and analyzing feature importance:

1. **plot_model_comparison.py** - Horizontal comparison of baseline models
2. **plot_rf_comparison.py** - Vertical comparison of Random Forest configurations
3. **plot_xgb_comparison.py** - Vertical comparison of XGBoost configurations
4. **plot_feature_importance.py** - Feature importance analysis with SHAP

## Quick Start

### Generate All Plots at Once

```bash
python plot/generate_all_plots.py
```

This will run all 4 scripts and generate all visualizations.

### Run Individual Scripts

```bash
# Model comparison
python plot/plot_model_comparison.py

# Random Forest comparison
python plot/plot_rf_comparison.py

# XGBoost comparison
python plot/plot_xgb_comparison.py

# Feature importance analysis
python plot/plot_feature_importance.py
```

## Generated Outputs

### 1. Model Comparison (`plot/model_comparison/`)

Compares baseline models: LogReg, Random Forest, XGBoost, SVM, and Neural Network (no SMOTE, no class weighting)

**Generated files:**
- `01_performance_metrics.png` - Accuracy, Balanced Accuracy, ROC AUC comparison
- `02_sensitivity_comparison.png` - Class-specific sensitivity for all models
- `03_pathological_performance.png` - Critical pathological class metrics
- `04_clinical_cost.png` - Clinical cost comparison
- `05_summary_table.png` - Comprehensive summary table

### 2. Random Forest Comparison (`plot/rf_comparison/`)

Compares 4 RF configurations:
- RF-Baseline: Clean data, no SMOTE, no class_weight
- RF-Normalized: Normalized data, no SMOTE, no class_weight
- RF-SMOTE: Normalized data, SMOTE, no class_weight
- RF-Full: Normalized data, SMOTE, class_weight=balanced

**Generated files:**
- `01_performance_metrics.png` - Performance across configurations
- `02_class_sensitivity.png` - Sensitivity for all classes
- `03_pathological_metrics.png` - Detailed pathological class metrics
- `04_clinical_cost.png` - Cost comparison
- `05_confusion_matrices.png` - All confusion matrices
- `06_summary_table.png` - Configuration comparison table

### 3. XGBoost Comparison (`plot/xgb_comparison/`)

Compares 3 XGBoost configurations:
- XGB-Baseline: Clean data, no SMOTE, no sample_weight
- XGB-Normalized: Normalized data, no SMOTE, no sample_weight
- XGB-Balanced: Clean data, no SMOTE, sample_weight=balanced

**Generated files:**
- `01_performance_metrics.png` - Performance across configurations
- `02_class_sensitivity.png` - Sensitivity for all classes
- `03_pathological_metrics.png` - Detailed pathological class metrics
- `04_clinical_cost.png` - Cost comparison
- `05_confusion_matrices.png` - All confusion matrices
- `06_performance_radar.png` - Radar chart comparison
- `07_summary_table.png` - Configuration comparison table

### 4. Feature Importance Analysis (`plot/feature_importance/`)

Analyzes feature importance from both model-based methods and SHAP analysis for RF and XGBoost

**Generated files:**
- `01_model_importance_comparison.png` - Side-by-side model importance
- `02_top5_features_comparison.png` - Top 5 features RF vs XGB
- `03_feature_importance_heatmap.png` - Heatmap across models
- `04_consensus_feature_ranking.png` - Consensus ranking
- `05_top5_summary_table.png` - Top 5 with clinical significance
- `06_shap_feature_importance.png` - SHAP overall importance
- `07_shap_class_specific.png` - SHAP class-specific analysis
- `consensus_feature_ranking.csv` - Consensus ranking data

## Requirements

All scripts use standard Python libraries:
- numpy
- pandas
- matplotlib
- seaborn
- Pillow (PIL)

These should already be installed in your environment.

## Data Sources

The scripts read from:
- `models/saved/*/training_summary.json` - Model training parameters and metrics
- `reports/new/evaluation/*/medical_metrics.json` - Medical evaluation metrics
- `models/saved/*/feature_importance.csv` - Model-based feature importance
- `reports/new/evaluation/*/shap/` - SHAP analysis visualizations

## Customization

### Modify Colors

Each script uses a consistent color scheme defined at the top. You can modify:
```python
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
```

### Adjust DPI

Change resolution by modifying:
```python
plt.rcParams['figure.dpi'] = 300  # Increase for higher quality
```

### Add More Models

To add more models to comparison, update the model dictionaries at the top of each script:
```python
MODELS = {
    'Model-Name': {
        'summary': 'path/to/training_summary.json',
        'medical': 'path/to/medical_metrics.json'
    }
}
```

## Notes

- All plots use English labels and descriptions
- High-resolution PNG format (300 DPI) suitable for presentations
- Color-blind friendly palettes where possible
- Consistent styling across all visualizations
- Clinical relevance emphasized in annotations

## Troubleshooting

**Issue:** Script can't find model files
- **Solution:** Ensure you've run the training scripts first and models exist in `models/saved/`

**Issue:** SHAP plots not displaying
- **Solution:** Verify SHAP analysis was run during training (only RF and XGB have SHAP)

**Issue:** Import errors
- **Solution:** Install missing packages: `pip install numpy pandas matplotlib seaborn pillow`

## Contact

For issues or questions about these visualization scripts, please refer to the main project documentation.
