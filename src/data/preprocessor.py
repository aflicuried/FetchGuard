from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# Domain ranges and assumptions for CTG tabular features (UCI CTG dataset)
CTG_RANGE_BOUNDS: Dict[str, Tuple[float, float]] = {
    # FHR baseline (bpm)
    "LB": (50.0, 240.0),
    # Percentages
    "ASTV": (0.0, 100.0),
    "ALTV": (0.0, 100.0),
    # STV/LTV (dataset-specific typical ranges; conservative)
    "MSTV": (0.0, 10.0),
    "MLTV": (0.0, 50.0),
    # Event counts (non-negative, conservative upper bounds)
    "AC": (0.0, 2000.0),
    "FM": (0.0, 5000.0),
    "UC": (0.0, 2000.0),
    "DL": (0.0, 2000.0),
    "DS": (0.0, 2000.0),
    "DP": (0.0, 2000.0),
    # Histogram-derived features (use broad numeric ranges)
    "Width": (0.0, 100.0),
    "Min": (0.0, 240.0),
    "Max": (0.0, 240.0),
    "Nmax": (0.0, 100.0),
    "Nzeros": (0.0, 100.0),
    "Mode": (0.0, 240.0),
    "Mean": (0.0, 240.0),
    "Median": (0.0, 240.0),
    "Variance": (0.0, 1e6),
    "Tendency": (-10.0, 10.0),
}


@dataclass
class QualityIssue:
    type: str
    details: Dict[str, object]


class CTGPreprocessor:
    """Preprocess CTG tabular features with medically-informed rules and auditing.

    Methods are pure-transform style where possible; each method logs decisions
    and returns transformed DataFrames along with optional reports for audit.
    """

    def __init__(self, target_column: str = "NSP") -> None:
        self.target_column = target_column

    # ---------- core transforms ----------
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        out = df.drop_duplicates()
        removed = before - len(out)
        if removed:
            logger.info("Removed %d duplicate rows (%.2f%%)", removed, 100.0 * removed / max(before, 1))
        return out

    def validate_medical_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip values into plausible medical ranges, logging the number of adjustments per column."""
        df = df.copy()
        for col, (low, high) in CTG_RANGE_BOUNDS.items():
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce")
                below = series.lt(low).sum()
                above = series.gt(high).sum()
                if below or above:
                    logger.warning(
                        "Column %s: %d below %.2f, %d above %.2f. Clipping to range.", col, int(below), low, int(above), high
                    )
                df[col] = series.clip(lower=low, upper=high)
        return df

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers based on hard medical limits beyond extreme physiology.

        For safety, this method removes rows where LB < 50 or LB > 240 bpm (sensor/entry errors),
        and where key percentage metrics are outside [0, 100].
        """
        df = df.copy()
        mask_valid = pd.Series(True, index=df.index)
        if "LB" in df:
            lb = pd.to_numeric(df["LB"], errors="coerce")
            mask_valid &= lb.between(50, 240, inclusive="both")
        for pct_col in ["ASTV", "ALTV"]:
            if pct_col in df:
                s = pd.to_numeric(df[pct_col], errors="coerce")
                mask_valid &= s.between(0, 100, inclusive="both")
        removed = (~mask_valid).sum()
        if removed:
            logger.info("Removed %d rows as medical outliers", int(removed))
        return df.loc[mask_valid].reset_index(drop=True)

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "median_by_class") -> pd.DataFrame:
        """Impute missing values using domain-informed strategies.

        - median_by_class: compute median per NSP class for numeric features; fallback to global median.
        - robust: winsorize extreme values then fill with median.
        - conservative_zero_for_counts: for event count columns (AC, UC, DL, DS, DP, FM) fill zeros, others median.
        """
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if strategy == "median_by_class" and self.target_column in df.columns:
            # per-class median for numeric features
            medians_global = df[numeric_cols].median(numeric_only=True)
            for cls, group in df.groupby(self.target_column):
                medians = group[numeric_cols].median(numeric_only=True)
                idx = group.index
                for col in numeric_cols:
                    df.loc[idx, col] = group[col].fillna(medians.get(col, medians_global.get(col)))
            logger.info("Missing values imputed by per-class medians with global fallback")

        elif strategy == "robust":
            # winsorize then median-impute
            for col in numeric_cols:
                s = pd.to_numeric(df[col], errors="coerce")
                q1, q3 = s.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                s_w = s.clip(lower=lower, upper=upper)
                df[col] = s_w.fillna(s_w.median())
            logger.info("Missing values imputed by robust winsorized median")

        elif strategy == "conservative_zero_for_counts":
            count_cols = [c for c in ["AC", "UC", "DL", "DS", "DP", "FM"] if c in df.columns]
            for col in count_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            other_cols = [c for c in numeric_cols if c not in count_cols]
            medians = df[other_cols].median(numeric_only=True)
            for col in other_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(medians[col])
            logger.info("Counts filled with 0, others with median")

        else:
            # default fallback: global medians, but log a warning to discourage naive use
            medians = df[numeric_cols].median(numeric_only=True)
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(medians)
            logger.warning("Fallback median imputation used (strategy=%s)", strategy)

        return df

    # ---------- quality checks ----------
    def quality_checks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run medical QA checks and return a table of flagged issues."""
        issues: List[QualityIssue] = []

        # 1) Baseline FHR normal range check 110-160 bpm (flag outside but physiological)
        if "LB" in df.columns:
            lb = pd.to_numeric(df["LB"], errors="coerce")
            outside = ~lb.between(110, 160, inclusive="both")
            count = int(outside.sum())
            if count:
                issues.append(QualityIssue(
                    type="baseline_fhr_outside_normal",
                    details={"rows": count, "percent": 100.0 * count / max(len(df), 1)}
                ))

        # 2) UC pattern plausibility: UC should be non-negative integers; excessive values flagged
        if "UC" in df.columns:
            uc = pd.to_numeric(df["UC"], errors="coerce")
            neg = int((uc < 0).sum())
            frac_non_int = float(((uc % 1) != 0).mean()) if uc.notna().any() else 0.0
            high = int((uc > 200).sum())
            if neg:
                issues.append(QualityIssue("uc_negative", {"rows": neg}))
            if frac_non_int > 0.05:
                issues.append(QualityIssue("uc_non_integer_ratio", {"ratio": frac_non_int}))
            if high:
                issues.append(QualityIssue("uc_suspiciously_high", {"rows": high}))

        # 3) Sensor error patterns: zero variability with non-zero events
        if set(["MSTV", "MLTV"]).issubset(df.columns):
            stv = pd.to_numeric(df["MSTV"], errors="coerce")
            ltv = pd.to_numeric(df["MLTV"], errors="coerce")
            zero_var = (stv.fillna(0) == 0) & (ltv.fillna(0) == 0)
            # while having accelerations/decelerations
            accel_cols = [c for c in ["AC", "DL", "DS", "DP"] if c in df.columns]
            if accel_cols:
                accel_sum = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce").fillna(0) for c in accel_cols}).sum(axis=1)
                suspicious = (zero_var) & (accel_sum > 0)
                count = int(suspicious.sum())
                if count:
                    issues.append(QualityIssue("sensor_inconsistency_zero_variability_with_events", {"rows": count}))

        # 4) Percentage constraints outside [0,100]
        for pct_col in ["ASTV", "ALTV"]:
            if pct_col in df.columns:
                s = pd.to_numeric(df[pct_col], errors="coerce")
                out = int((s < 0).sum() + (s > 100).sum())
                if out:
                    issues.append(QualityIssue("percentage_out_of_bounds", {"column": pct_col, "rows": out}))

        # 5) Constant features
        numeric_df = df.select_dtypes(include=[np.number])
        for col in numeric_df.columns:
            if col == self.target_column:
                continue
            if numeric_df[col].nunique(dropna=True) <= 1:
                issues.append(QualityIssue("constant_feature", {"column": col}))

        # Build report DataFrame
        report = pd.DataFrame([{"type": it.type, **it.details} for it in issues])
        return report

    # ---------- pipeline ----------
    def run(self, df: pd.DataFrame, missing_strategy: str = "median_by_class") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run full preprocessing pipeline and return (clean_df, quality_report)."""
        logger.info("Starting CTG preprocessing: %d rows, %d cols", df.shape[0], df.shape[1])

        df1 = self.remove_duplicates(df)
        df2 = self.remove_outliers(df1)
        df3 = self.validate_medical_ranges(df2)
        df4 = self.handle_missing_values(df3, strategy=missing_strategy)

        report = self.quality_checks(df4)
        logger.info("Completed CTG preprocessing: %d rows, %d cols", df4.shape[0], df4.shape[1])
        return df4, report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # Example manual run (expects prior data load)
    # from src.data.data_loader import DataLoader
    # loader = DataLoader()
    # df = loader.clean_ctg_dataframe(loader.load(sheet="Data"))
    # pre = CTGPreprocessor()
    # clean, qa = pre.run(df)
    # clean.to_csv("data/processed/ctg_clean.csv", index=False)
    # qa.to_csv("reports/preprocess_quality_report.csv", index=False)


