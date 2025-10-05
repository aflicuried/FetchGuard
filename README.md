# FetalGuard 🤰👶

**Automated Fetal Health Classification Using Cardiotocography (CTG) Data**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-blue)](https://mlflow.org/)

A comprehensive machine learning system for classifying fetal health status from Cardiotocography (CTG) recordings. This project achieves **95.56% accuracy** and **93.14% pathological sensitivity** using XGBoost, meeting clinical deployment standards.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Training Models](#training-models)
  - [Generating Visualizations](#generating-visualizations)
  - [Viewing Results](#viewing-results)
- [Model Performance](#model-performance)
- [Documentation](#documentation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## 🎯 Overview

### Problem Statement

Cardiotocography (CTG) is a critical tool for monitoring fetal health during pregnancy and labor. However, manual CTG interpretation is:
- **Subjective**: High inter-observer variability
- **Time-consuming**: Requires continuous expert attention
- **Error-prone**: Fatigue and cognitive load affect accuracy

### Our Solution

FetalGuard provides an automated, objective, and interpretable AI system for CTG classification into three categories:

| Class | Risk Level | Clinical Action |
|-------|------------|-----------------|
| **Normal** | Low | Routine monitoring |
| **Suspect** | Moderate | Increased surveillance |
| **Pathological** | High | Immediate intervention |

### Why It Matters

- **Patient Safety**: 93.14% sensitivity for detecting pathological cases
- **Clinical Trust**: Explainable AI with SHAP analysis
- **Evidence-Based**: Features align with medical guidelines (FIGO, NICE, ACOG)
- **Practical**: Fast inference (<10ms per prediction), CPU-only

---

## ✨ Key Features

### 🎯 State-of-the-Art Performance
- **95.56% Overall Accuracy**
- **93.14% Pathological Sensitivity** (critical for patient safety)
- **90.51% Balanced Accuracy** (handles class imbalance)
- **Clinical Cost: 284** (lowest among all models tested)

### 🔬 Comprehensive Model Evaluation
- **5 Algorithms Tested**: LogReg, Random Forest, XGBoost, SVM, Neural Network
- **SMOTE & Class Weighting Analysis**: Systematic evaluation of data augmentation techniques
- **Cross-Validation**: 5-fold stratified CV for robust evaluation
- **Medical Metrics**: Sensitivity, specificity, PPV, NPV for each class

### 📊 Rich Visualizations
- **25 High-Quality Plots**: All presentation-ready at 300 DPI
- **Horizontal Comparison**: All models side-by-side
- **Vertical Analysis**: Configuration optimization for RF and XGBoost
- **Feature Importance**: Model-based + SHAP analysis

### 🧠 Explainable AI
- **SHAP Analysis**: Understand individual prediction reasoning
- **Feature Importance**: Identify key predictors (Mean FHR, MSTV, ALTV, ASTV, DP)
- **Clinical Validation**: Features align with medical literature
- **Transparency**: Clear documentation of model decisions

### 🔄 MLflow Integration
- **Experiment Tracking**: All training runs logged automatically
- **Model Versioning**: Easy comparison across experiments
- **Reproducibility**: Full parameter and metric tracking
- **Artifact Management**: Models, plots, and metadata stored systematically

---

## 📁 Project Structure

```
FetalGuard/
│
├── README.md                      # This file
├── PROJECT_REPORT.md              # Comprehensive analysis report
├── requirements.txt               # Python dependencies
│
├── data/
│   ├── raw/
│   │   └── CTG.xls               # Original dataset
│   └── processed/
│       ├── ctg_clean.csv         # Cleaned data
│       └── ctg_normalized.csv    # Normalized data
│
├── src/
│   ├── data/
│   │   ├── create_correlation_heatmap.py  # Correlation analysis
│   │   └── normalize_data.py              # Data normalization
│   │
│   └── models/
│       ├── train.py              # Main training script (LogReg, RF, XGB, SVM)
│       ├── train_nn.py           # Neural network training
│       └── evaluate.py           # Model evaluation (deprecated)
│
├── models/
│   ├── saved/                    # Trained models
│   │   ├── xgb_YYYYMMDD_HHMMSS/
│   │   │   ├── model.joblib
│   │   │   ├── training_summary.json
│   │   │   └── feature_importance.csv
│   │   └── ...
│   │
│   └── mlruns/                   # MLflow experiment tracking
│       └── [experiment_id]/
│           └── [run_id]/
│               ├── meta.yaml
│               ├── metrics/
│               ├── params/
│               └── artifacts/
│
├── reports/
│   └── new/
│       └── evaluation/           # Evaluation results
│           ├── xgb_YYYYMMDD_HHMMSS/
│           │   ├── medical_evaluation_report.txt
│           │   ├── medical_metrics.json
│           │   ├── confusion_matrix_cv.png
│           │   ├── roc_curves_cv.png
│           │   └── shap/
│           │       ├── shap_feature_importance.png
│           │       ├── shap_summary_class_0_normal.png
│           │       ├── shap_summary_class_1_suspect.png
│           │       └── shap_summary_class_2_pathological.png
│           └── ...
│
└── plot/                         # Visualization scripts and outputs
    ├── generate_all_plots.py     # Master script
    ├── plot_model_comparison.py  # Baseline model comparison
    ├── plot_rf_comparison.py     # Random Forest analysis
    ├── plot_xgb_comparison.py    # XGBoost analysis
    ├── plot_feature_importance.py # Feature importance
    ├── README.md                 # Plotting documentation
    ├── VISUALIZATION_SUMMARY.md  # Detailed analysis
    ├── RESULTS.txt               # Summary results
    │
    ├── model_comparison/         # 5 plots
    ├── rf_comparison/            # 6 plots
    ├── xgb_comparison/           # 7 plots
    └── feature_importance/       # 7 plots + 1 CSV
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) Virtual environment tool (venv, conda)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/FetalGuard.git
cd FetalGuard
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Key Dependencies

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
imbalanced-learn>=0.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
mlflow>=2.9.2
shap>=0.42.0
joblib>=1.3.0
torch>=2.0.0  # For neural network
```

---

## ⚡ Quick Start

### 1. Prepare Data

```bash
# Data should be placed in data/raw/CTG.xls
# Run preprocessing (if needed)
python src/data/normalize_data.py
```

### 2. Train Best Model (XGBoost)

```bash
python src/models/train.py --algo xgb --cv 5 --random-state 42
```

Expected output:
```
Training XGBoost with 5-fold cross-validation...
CV Accuracy: 0.9556
CV Balanced Accuracy: 0.9051
Pathological Sensitivity: 0.9314
Model saved to: models/saved/xgb_20251005_HHMMSS/
```

### 3. Generate All Visualizations

```bash
python plot/generate_all_plots.py
```

This creates 25 plots in `plot/` subdirectories.

### 4. View Results

- **Comprehensive Report**: Open `PROJECT_REPORT.md`
- **Model Summary**: Check `models/saved/xgb_*/training_summary.json`
- **Evaluation Metrics**: Review `reports/new/evaluation/xgb_*/medical_metrics.json`
- **Visualizations**: Browse `plot/*/` directories

---

## 📖 Usage

### Training Models

#### XGBoost (Recommended)

```bash
# Baseline (best performance)
python src/models/train.py --algo xgb

# With sample weighting
python src/models/train.py --algo xgb --class-weight balanced

# With normalized data
python src/models/train.py --algo xgb --use-normalized
```

#### Random Forest

```bash
# Baseline
python src/models/train.py --algo rf

# Full configuration (SMOTE + class weighting)
python src/models/train.py --algo rf --use-normalized --use-smote --class-weight balanced
```

#### Other Models

```bash
# Logistic Regression
python src/models/train.py --algo logreg

# Support Vector Machine
python src/models/train.py --algo svm

# LightGBM
python src/models/train.py --algo lgbm
```

#### Neural Network

```bash
# Default architecture [256, 128, 64]
python src/models/train_nn.py

# Custom architecture
python src/models/train_nn.py --hidden-dims 128 64 32 --dropout 0.3 --lr 0.001 --epochs 200
```

### Command-Line Arguments

#### train.py

```
--algo              Algorithm: xgb, rf, logreg, svm, lgbm
--use-smote         Apply SMOTE for class balancing
--class-weight      Class weighting: None or 'balanced'
--cv                Number of CV folds (default: 5)
--random-state      Random seed (default: 42)
--use-normalized    Use normalized data instead of clean data
```

#### train_nn.py

```
--hidden-dims       Hidden layer dimensions (default: 256 128 64)
--dropout           Dropout rate (default: 0.2)
--lr                Learning rate (default: 0.001)
--weight-decay      L2 regularization (default: 0.01)
--batch-size        Batch size (default: 64)
--epochs            Max epochs (default: 200)
--patience          Early stopping patience (default: 15)
--use-normalized    Use normalized data
```

### Generating Visualizations

```bash
# Generate all plots
python plot/generate_all_plots.py

# Or generate individually
python plot/plot_model_comparison.py     # Baseline comparison
python plot/plot_rf_comparison.py        # Random Forest analysis
python plot/plot_xgb_comparison.py       # XGBoost analysis
python plot/plot_feature_importance.py   # Feature importance
```

### Viewing MLflow Experiments

```bash
# Start MLflow UI
mlflow ui --backend-store-uri file:./models/mlruns

# Open browser to http://localhost:5000
```

### Viewing Results

#### Training Summary (JSON)

```bash
cat models/saved/xgb_20251005_HHMMSS/training_summary.json
```

Example:
```json
{
  "timestamp": "20251005_171512",
  "parameters": {
    "algo": "xgb",
    "use_smote": false,
    "class_weight": null,
    "cv": 5
  },
  "metrics": {
    "acc": 0.9556,
    "bacc": 0.9051,
    "f1_macro": 0.9208,
    "roc_auc_ovr": 0.9883
  }
}
```

#### Medical Evaluation Report (Text)

```bash
cat reports/new/evaluation/xgb_20251005_HHMMSS/medical_evaluation_report.txt
```

Example:
```
=== MEDICAL EVALUATION REPORT ===
Model: XGBoost_20251005_171512_CV

CRITICAL PATHOLOGICAL CLASS PERFORMANCE:
- Sensitivity: 0.931
- Specificity: 0.996
- PPV: 0.953
- NPV: 0.994

PATIENT SAFETY ASSESSMENT:
- Safety Level: EXCELLENT
- Comment: Very low risk of missing pathological cases
```

---

## 📊 Model Performance

### Comparison of All Models

| Model | Accuracy | Balanced Accuracy | Pathological Sensitivity | Clinical Cost |
|-------|----------|-------------------|--------------------------|---------------|
| **XGBoost** ⭐ | **95.56%** | **90.51%** | **93.14%** | **284** |
| Random Forest (Full) | 93.95% | 89.66% | 88.00% | 468 |
| Random Forest (Baseline) | 93.90% | 86.39% | 88.00% | 468 |
| Neural Network | 92.10% | 81.66% | 79.43% | 655 |
| Logistic Regression | 89.36% | 78.52% | 76.00% | 833 |
| SVM | 84.54% | 85.10% | 78.00% | 860 |

⭐ **Recommended for clinical deployment**

### XGBoost Confusion Matrix

```
                  Predicted
Actual      Normal  Suspect  Pathological
Normal        1625      20            3
Suspect         54     233            5
Pathological     6       6          163
```

**Key Metrics**:
- **False Negative Rate (Pathological)**: 6.9% (12/175)
- **True Positive Rate (Pathological)**: 93.1% (163/175)
- **Overall Accuracy**: 95.6% (2021/2115)

### Impact of SMOTE and Class Weighting

#### Random Forest Results

| Configuration | Accuracy | Balanced Acc | Pathological Sensitivity |
|---------------|----------|--------------|--------------------------|
| Baseline | 93.90% | 86.39% | 88.00% |
| + Normalization | 93.95% | 86.59% | 88.00% |
| + SMOTE | 94.23% | **89.96%** | 88.00% |
| + SMOTE + Class Weight | 93.95% | **89.66%** | 88.00% |

**Improvement**: SMOTE alone improves balanced accuracy by +3.57% while maintaining pathological sensitivity.

#### XGBoost Results

| Configuration | Accuracy | Balanced Acc | Pathological Sensitivity |
|---------------|----------|--------------|--------------------------|
| Baseline | **95.56%** | 90.51% | **93.14%** |
| + Normalization | **95.56%** | 90.51% | **93.14%** |
| + Sample Weight | 95.13% | **91.68%** | 92.00% |

**Observation**: XGBoost is robust to preprocessing; baseline configuration performs best.

---

## 📚 Documentation

### Main Documents

1. **[PROJECT_REPORT.md](PROJECT_REPORT.md)** - Comprehensive analysis with all visualizations
   - Model performance comparison
   - SMOTE and class weighting analysis
   - Medical feature interpretation
   - Clinical validation and recommendations

2. **[plot/README.md](plot/README.md)** - Visualization guide
   - How to generate plots
   - Description of each visualization
   - Customization options

3. **[plot/VISUALIZATION_SUMMARY.md](plot/VISUALIZATION_SUMMARY.md)** - Detailed analysis
   - Key findings from each plot
   - Clinical implications
   - Recommended plots for presentations

4. **[plot/RESULTS.txt](plot/RESULTS.txt)** - Quick reference
   - Best model performance
   - Top 5 features
   - Recommended plots for slides

### Code Documentation

All Python scripts include:
- Docstrings for functions and classes
- Inline comments explaining logic
- Type hints for parameters
- Usage examples in headers

### Medical Background

For clinical context, see:
- [FIGO Guidelines (2015)](https://obgyn.onlinelibrary.wiley.com/doi/10.1111/1471-0528.13108)
- [NICE Guidelines (2017)](https://www.nice.org.uk/guidance/cg190)
- [ACOG Practice Bulletin (2010)](https://www.acog.org/clinical/clinical-guidance/practice-bulletin/articles/2010/07/intrapartum-fetal-heart-rate-monitoring)

---

## 🎯 Results

### Top 5 Predictive Features

| Rank | Feature | Importance | Clinical Significance |
|------|---------|------------|----------------------|
| 1 | **Mean FHR** | 0.1153 | Baseline heart rate; normal: 110-160 bpm |
| 2 | **MSTV** | 0.1143 | Short-term variability; loss indicates hypoxia |
| 3 | **ALTV** | 0.1092 | Long-term variability abnormalities |
| 4 | **ASTV** | 0.1017 | Short-term variability abnormalities |
| 5 | **DP** | 0.0893 | Prolonged decelerations; critical warning sign |

**Clinical Validation**: These features align with established CTG interpretation guidelines (FIGO, NICE, ACOG), supporting model validity.

### Model Interpretability

- **SHAP Analysis**: Available for Random Forest and XGBoost
- **Feature Importance**: Consistent across models
- **Class-Specific Explanations**: SHAP shows how features affect each class prediction
- **Individual Predictions**: Can explain why specific CTG was classified as pathological

### Visualizations

All 25 plots available in `plot/` directory:
- 📊 **5 plots**: Baseline model comparison
- 🌲 **6 plots**: Random Forest configuration analysis
- 🚀 **7 plots**: XGBoost configuration analysis
- 🔍 **7 plots**: Feature importance and SHAP analysis

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### Reporting Issues

- Use GitHub Issues
- Provide clear description
- Include error messages and logs
- Specify Python version and OS

### Pull Requests

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

### Code Standards

- Follow PEP 8 style guide
- Add docstrings to functions
- Include type hints
- Write unit tests for new features
- Update documentation

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Important Note

**This software is for research and educational purposes only.**

Clinical deployment requires:
- ✅ Regulatory approval (FDA, CE marking, etc.)
- ✅ Validation on local patient populations
- ✅ Integration with hospital systems
- ✅ Physician oversight and verification
- ✅ Regular performance monitoring
- ✅ Compliance with medical device regulations

**FetalGuard is NOT a substitute for professional medical judgment.**

---

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@software{fetalguard2025,
  title = {FetalGuard: Automated Fetal Health Classification Using CTG Data},
  author = {[Your Name]},
  year = {2025},
  url = {https://github.com/yourusername/FetalGuard},
  note = {Machine learning system for CTG classification achieving 95.56\% accuracy}
}
```

### Dataset Citation

```bibtex
@misc{ctg_dataset,
  title = {Cardiotocography Data Set},
  author = {Marques de S{\'a}, J.P. and Bernardes, J. and Ayres de Campos, D.},
  year = {2010},
  howpublished = {UCI Machine Learning Repository},
  url = {https://archive.ics.uci.edu/ml/datasets/cardiotocography}
}
```

---

## 👥 Authors

- **[Your Name]** - *Initial work* - [GitHub Profile](https://github.com/yourusername)

---

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for providing the CTG dataset
- **Medical advisors** for clinical validation guidance
- **Open-source community** for excellent ML libraries (scikit-learn, XGBoost, SHAP)
- **MLflow team** for experiment tracking infrastructure

---

## 📞 Contact

For questions or collaboration:

- **Email**: your.email@example.com
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/FetalGuard/issues)
- **Twitter**: [@yourhandle](https://twitter.com/yourhandle)

---

## 🗺️ Roadmap

### Short-term (Next 3 months)
- [ ] Add temporal analysis for trend detection
- [ ] Develop web interface for demo
- [ ] Create Docker container for easy deployment
- [ ] Add more neural network architectures (LSTM, Transformer)

### Medium-term (6-12 months)
- [ ] External validation on independent datasets
- [ ] Integration with hospital EHR systems
- [ ] Real-time monitoring dashboard
- [ ] Mobile app for point-of-care use

### Long-term (1+ years)
- [ ] Multi-center clinical trial
- [ ] Regulatory approval process
- [ ] Commercial deployment
- [ ] Integration with wearable CTG devices

---

## 📊 Project Status

- ✅ **Data preprocessing**: Complete
- ✅ **Model training**: Complete (5 algorithms)
- ✅ **Model evaluation**: Complete (comprehensive metrics)
- ✅ **Visualization**: Complete (25 plots)
- ✅ **Documentation**: Complete
- 🔄 **Clinical validation**: In progress
- ⏳ **Regulatory approval**: Not started
- ⏳ **Deployment**: Not started

---

## ⚠️ Disclaimer

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

**Medical Disclaimer**: This software is not intended to diagnose, treat, cure, or prevent any disease. Always consult with qualified healthcare professionals for medical advice.

---

<p align="center">
  Made with ❤️ for better maternal and fetal health outcomes
</p>

<p align="center">
  <a href="#fetalguard-">Back to top</a>
</p>
