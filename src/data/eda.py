from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.data_loader import DataLoader


logger = logging.getLogger(__name__)


MEDICAL_FEATURE_CONTEXT = {
    # Common UCI CTG features; adjust as needed for your dataset schema
    "LB": "FHR baseline (bpm)",
    "AC": "Number of accelerations",
    "FM": "Number of fetal movements",
    "UC": "Number of uterine contractions",
    "DL": "Light decelerations",
    "DS": "Severe decelerations",
    "DP": "Prolonged decelerations",
    "ASTV": "Percentage of abnormal short-term variability",
    "MSTV": "Mean value of short-term variability",
    "ALTV": "Percentage of abnormal long-term variability",
    "MLTV": "Mean value of long-term variability",
    "Width": "Histogram width",
    "Min": "Histogram minimum",
    "Max": "Histogram maximum",
    "Nmax": "Number of histogram peaks",
    "Nzeros": "Number of histogram zeros",
    "Mode": "Histogram mode",
    "Mean": "Histogram mean",
    "Median": "Histogram median",
    "Variance": "Histogram variance",
    "Tendency": "Histogram tendency",
}


def ensure_reports_dir(reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)


def describe_with_context(df: pd.DataFrame) -> pd.DataFrame:
    desc = df.describe(include=[np.number]).T
    desc["medical_context"] = [MEDICAL_FEATURE_CONTEXT.get(c, "") for c in desc.index]
    return desc


def class_distribution(df: pd.DataFrame, target_col: str = "NSP") -> pd.Series:
    if target_col not in df:
        raise KeyError(f"Missing target column {target_col}")
    counts = df[target_col].value_counts().sort_index()
    counts.index = counts.index.map({1: "Normal", 2: "Suspect", 3: "Pathologic"})
    return counts


def correlation_focus(df: pd.DataFrame, reports_dir: Path) -> None:
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=[c for c in ["NSP"] if c in df], errors="ignore")
    corr = numeric_df.corr(numeric_only=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=False)
    plt.title("Correlation matrix of CTG features")
    plt.tight_layout()
    outfile = reports_dir / "correlation_matrix.png"
    plt.savefig(outfile, dpi=200)
    plt.close()
    logger.info("Saved correlation matrix to %s", outfile)


def plot_distributions(df: pd.DataFrame, reports_dir: Path) -> None:
    key_features = [
        ("LB", "baseline_fhr"),
        ("AC", "accelerations"),
        ("DL", "light_decelerations"),
        ("DS", "severe_decelerations"),
        ("DP", "prolonged_decelerations"),
        ("MSTV", "short_term_variability"),
        ("MLTV", "long_term_variability"),
        ("ASTV", "abnormal_stv_percent"),
        ("ALTV", "abnormal_ltv_percent"),
    ]
    for col, name in key_features:
        if col not in df.columns:
            logger.warning("Missing expected feature for distribution plot: %s", col)
            continue
        plt.figure(figsize=(7, 4))
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.xlabel(f"{col} - {MEDICAL_FEATURE_CONTEXT.get(col, '')}")
        plt.ylabel("Count")
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        outfile = reports_dir / f"dist_{name}.png"
        plt.savefig(outfile, dpi=200)
        plt.close()
        logger.info("Saved distribution plot for %s to %s", col, outfile)


def detect_data_quality_issues(df: pd.DataFrame) -> pd.DataFrame:
    issues = []
    # 1) Missing values
    missing = df.isna().sum()
    for col, cnt in missing.items():
        if cnt > 0:
            issues.append({"type": "missing_values", "column": col, "count": int(cnt)})

    # 2) Physiological plausibility checks (heuristic ranges)
    plausible_ranges = {
        "LB": (60, 220),  # bpm
        "ASTV": (0, 100),
        "ALTV": (0, 100),
        "MSTV": (0, 10),
        "MLTV": (0, 50),
        "AC": (0, 1000),
        "DL": (0, 1000),
        "DS": (0, 1000),
        "DP": (0, 1000),
        "UC": (0, 1000),
    }
    for col, (low, high) in plausible_ranges.items():
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            out_low = series[series < low].count()
            out_high = series[series > high].count()
            if out_low + out_high > 0:
                issues.append({
                    "type": "out_of_range",
                    "column": col,
                    "below": int(out_low),
                    "above": int(out_high),
                    "range": f"[{low},{high}]",
                })

    # 3) Constant or near-constant features
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    for col in numeric_df.columns:
        if col == "NSP":
            continue
        unique_vals = numeric_df[col].nunique(dropna=True)
        if unique_vals <= 1:
            issues.append({"type": "constant_feature", "column": col})

    return pd.DataFrame(issues)


def run_eda(raw_dir: Path = Path("data/raw"), reports_dir: Path = Path("reports")) -> None:
    ensure_reports_dir(reports_dir)
    loader = DataLoader(raw_dir)
    # Force using 'Data' sheet for CTG Excel and clean separators/unnamed columns
    df = loader.load(sheet="Data")
    df = loader.clean_ctg_dataframe(df)

    # Save a quick columns preview to help inspection
    preview = pd.DataFrame({
        "column": df.columns,
        "non_null": [df[c].notna().sum() for c in df.columns],
        "dtype": [str(df[c].dtype) for c in df.columns],
    })
    preview.to_csv(reports_dir / "columns_preview.csv", index=False)

    # Summary with medical context
    desc = describe_with_context(df)
    desc_path = reports_dir / "summary_with_medical_context.csv"
    desc.to_csv(desc_path)
    logger.info("Saved statistical summary to %s", desc_path)

    # Class distribution
    try:
        counts = class_distribution(df)
        counts_path = reports_dir / "class_distribution.csv"
        counts.to_csv(counts_path, header=["count"]) \
            if hasattr(counts, "to_csv") else counts.to_frame("count").to_csv(counts_path)
        plt.figure(figsize=(6, 4))
        sns.barplot(x=counts.index, y=counts.values)
        plt.title("Class distribution (NSP)")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.tight_layout()
        outfile = reports_dir / "class_distribution.png"
        plt.savefig(outfile, dpi=200)
        plt.close()
        logger.info("Saved class distribution plot to %s", outfile)
    except Exception as exc:
        logger.warning("Class distribution unavailable: %s", exc)

    # Correlation matrix (after removing constant/near-constant columns)
    try:
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        # drop constant columns
        constant_cols = [c for c in numeric_df.columns if c != "NSP" and numeric_df[c].nunique(dropna=True) <= 1]
        numeric_df = numeric_df.drop(columns=constant_cols, errors="ignore")
        if not numeric_df.empty:
            correlation_focus(pd.concat([numeric_df, df[[c for c in ["NSP"] if c in df]]], axis=1), reports_dir)
    except Exception as exc:
        logger.warning("Failed to create correlation matrix: %s", exc)

    # Distributions for key features
    try:
        plot_distributions(df, reports_dir)
    except Exception as exc:
        logger.warning("Failed to create distributions: %s", exc)

    # Data quality issues
    issues = detect_data_quality_issues(df)
    issues_path = reports_dir / "data_quality_issues.csv"
    issues.to_csv(issues_path, index=False)
    logger.info("Saved data quality issues to %s", issues_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_eda()


