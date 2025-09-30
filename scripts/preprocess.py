from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import DataLoader
from src.data.preprocessor import CTGPreprocessor
from src.data.validator import CTGValidator
from src.features.feature_engineer import FeatureEngineer
from src.features.feature_selector import FeatureSelector


def main() -> None:
    parser = argparse.ArgumentParser(description="CTG preprocessing pipeline")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Directory containing CTG files")
    parser.add_argument("--sheet", type=str, default="Data", help="Excel sheet name to load")
    parser.add_argument("--missing-strategy", type=str, default="median_by_class", choices=[
        "median_by_class", "robust", "conservative_zero_for_counts"
    ], help="Missing value handling strategy")
    parser.add_argument("--out-data", type=Path, default=Path("data/processed/ctg_clean.csv"), help="Output CSV for cleaned data")
    parser.add_argument("--out-report", type=Path, default=Path("reports/preprocess_quality_report.csv"), help="Output CSV for QA report")
    parser.add_argument("--out-validate", type=Path, default=Path("reports/validation_report.csv"), help="Output CSV for validation report")
    # Feature engineering / selection outputs
    parser.add_argument("--out-fe-data", type=Path, default=Path("data/processed/ctg_with_features.csv"), help="Output CSV for engineered features")
    parser.add_argument("--out-fe-meta", type=Path, default=Path("reports/feature_metadata.csv"), help="Output CSV for feature metadata")
    parser.add_argument("--out-fs-scores", type=Path, default=Path("reports/feature_selection_scores.csv"), help="Output CSV for selection scores")
    parser.add_argument("--out-fs-plot", type=Path, default=Path("reports/feature_importance.png"), help="Output image for importance plot")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K features to select (protected features always kept)")
    parser.add_argument("--target-column", type=str, default="NSP", help="Target column name for feature selection")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s - %(levelname)s - %(message)s")

    loader = DataLoader(args.raw_dir)
    df = loader.clean_ctg_dataframe(loader.load(sheet=args.sheet))

    pre = CTGPreprocessor()
    clean_df, qa = pre.run(df, missing_strategy=args.missing_strategy)

    args.out_data.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(args.out_data, index=False)
    qa.to_csv(args.out_report, index=False)

    # Run validation report on cleaned data
    validator = CTGValidator()
    val_report = validator.quality_report(clean_df)
    val_report.to_csv(args.out_validate, index=False)

    # Feature engineering
    fe = FeatureEngineer()
    fe_data, fe_meta = fe.transform(clean_df)
    args.out_fe_data.parent.mkdir(parents=True, exist_ok=True)
    args.out_fe_meta.parent.mkdir(parents=True, exist_ok=True)
    fe_data.to_csv(args.out_fe_data, index=False)
    fe_meta.to_csv(args.out_fe_meta, index=False)

    # Feature selection
    # Ensure target column is present in engineered data
    target_col = args.target_column
    if target_col not in fe_data.columns and target_col in clean_df.columns:
        fe_data[target_col] = clean_df[target_col]
        logging.info("Injected target column %s into engineered dataset", target_col)

    fs = FeatureSelector(target_column=target_col)
    result = fs.select_top_k_features(fe_data, k=args.top_k)
    args.out_fs_scores.parent.mkdir(parents=True, exist_ok=True)
    result.scores.to_csv(args.out_fs_scores)
    # Importance plot
    fs.plot_importance(result.scores, top_n=args.top_k, outfile=args.out_fs_plot)

    logging.info(
        "Saved cleaned=%s, QA=%s, validation=%s, engineered=%s, meta=%s, selection_scores=%s, importance_plot=%s",
        args.out_data,
        args.out_report,
        args.out_validate,
        args.out_fe_data,
        args.out_fe_meta,
        args.out_fs_scores,
        args.out_fs_plot,
    )


if __name__ == "__main__":
    main()


