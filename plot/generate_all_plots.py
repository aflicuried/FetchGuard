"""
Master Script to Generate All Presentation Plots
Run this script to generate all visualizations for the presentation
"""
import subprocess
import sys
from pathlib import Path

def run_script(script_name):
    """Run a plotting script"""
    print("\n" + "="*70)
    print(f"Running: {script_name}")
    print("="*70)

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print(f"[SUCCESS] {script_name} completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error running {script_name}:")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise

def main():
    """Generate all plots"""
    print("="*70)
    print("GENERATING ALL PRESENTATION PLOTS")
    print("="*70)
    print("\nThis will create:")
    print("  1. Baseline model comparison (LogReg, RF, XGB, SVM, NN)")
    print("  2. Random Forest configuration comparison")
    print("  3. XGBoost configuration comparison")
    print("  4. Feature importance analysis (RF & XGB)")
    print("\n" + "="*70)

    # Get script directory
    script_dir = Path(__file__).parent

    # List of scripts to run
    scripts = [
        script_dir / "plot_model_comparison.py",
        script_dir / "plot_rf_comparison.py",
        script_dir / "plot_xgb_comparison.py",
        script_dir / "plot_feature_importance.py"
    ]

    # Run each script
    for script in scripts:
        if script.exists():
            run_script(script)
        else:
            print(f"[ERROR] Script not found: {script}")

    print("\n" + "="*70)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*70)
    print("\nOutput directories:")
    print("  - plot/model_comparison/")
    print("  - plot/rf_comparison/")
    print("  - plot/xgb_comparison/")
    print("  - plot/feature_importance/")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
