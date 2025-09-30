from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


CTG_GUIDELINES_CITATION = (
    "CTG interpretation principles per standard obstetric guidelines: baseline 110–160 bpm, "
    "assessment of variability, accelerations, and decelerations; context-dependent clinical correlation required."
)


@dataclass
class ValidationIssue:
    type: str
    details: Dict[str, object]


class CTGValidator:
    """Validate CTG tabular features with medical-rule checks and produce a quality report.

    The dataset is aggregated-per-record CTG features (UCI CTG style). Some checks use
    optional columns if present (e.g., duration). All checks log decisions for audit.
    """

    def __init__(self, target_column: str = "NSP") -> None:
        self.target_column = target_column

    # ---------- Checks ----------
    def check_time_consistency(
        self,
        df: pd.DataFrame,
        expected_sampling_hz: int = 4,
        min_duration_min: int = 10,
        duration_column: Optional[str] = None,
    ) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        # If a duration column is provided, ensure minimal duration
        if duration_column and duration_column in df.columns:
            dur = pd.to_numeric(df[duration_column], errors="coerce")
            too_short = (dur < min_duration_min).fillna(True)
            count = int(too_short.sum())
            if count:
                issues.append(ValidationIssue(
                    type="duration_below_min",
                    details={"rows": count, "min_required_min": min_duration_min}
                ))
        else:
            # Without explicit duration, flag as unknown; many UCI CTG records are fixed-length
            issues.append(ValidationIssue(
                type="duration_unknown",
                details={"assumed_sampling_hz": expected_sampling_hz, "min_required_min": min_duration_min}
            ))

        return issues

    def check_physiological_plausibility(self, df: pd.DataFrame) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        # Baseline reasonable physiological bounds
        if "LB" in df:
            lb = pd.to_numeric(df["LB"], errors="coerce")
            out = (~lb.between(50, 240, inclusive="both")).sum()
            if out:
                issues.append(ValidationIssue("lb_out_of_physiology", {"rows": int(out)}))

        # Normal baseline band 110–160 (not invalid, but noteworthy)
        if "LB" in df:
            lb = pd.to_numeric(df["LB"], errors="coerce")
            outside = (~lb.between(110, 160, inclusive="both")).sum()
            if outside:
                issues.append(ValidationIssue("lb_outside_normal_band", {"rows": int(outside)}))

        # Percentages must be in [0,100]
        for pct_col in ["ASTV", "ALTV"]:
            if pct_col in df:
                s = pd.to_numeric(df[pct_col], errors="coerce")
                out = int((s < 0).sum() + (s > 100).sum())
                if out:
                    issues.append(ValidationIssue("percentage_out_of_bounds", {"column": pct_col, "rows": out}))

        # Zero variability with events is suspicious (sensor issue)
        if set(["MSTV", "MLTV"]).issubset(df.columns):
            stv = pd.to_numeric(df["MSTV"], errors="coerce").fillna(0)
            ltv = pd.to_numeric(df["MLTV"], errors="coerce").fillna(0)
            zero_var = (stv == 0) & (ltv == 0)
            accel_cols = [c for c in ["AC", "DL", "DS", "DP"] if c in df.columns]
            if accel_cols:
                accel = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce").fillna(0) for c in accel_cols}).sum(axis=1)
                suspicious = (zero_var) & (accel > 0)
                if int(suspicious.sum()) > 0:
                    issues.append(ValidationIssue("sensor_inconsistency_zero_var_with_events", {"rows": int(suspicious.sum())}))

        # Histogram consistency: Min <= Mode <= Max and LB within [Min, Max] when available
        if set(["Min", "Mode", "Max"]).issubset(df.columns):
            mn = pd.to_numeric(df["Min"], errors="coerce")
            md = pd.to_numeric(df["Mode"], errors="coerce")
            mx = pd.to_numeric(df["Max"], errors="coerce")
            bad_hist = (mn > md) | (md > mx)
            if int(bad_hist.sum()) > 0:
                issues.append(ValidationIssue("histogram_inconsistent", {"rows": int(bad_hist.sum())}))
            if "LB" in df.columns:
                lb = pd.to_numeric(df["LB"], errors="coerce")
                lb_outside_hist = ~lb.between(mn, mx)
                if int(lb_outside_hist.sum()) > 0:
                    issues.append(ValidationIssue("lb_outside_histogram_range", {"rows": int(lb_outside_hist.sum())}))

        # UC plausibility: non-negative, integer-like, not excessively high
        if "UC" in df.columns:
            uc = pd.to_numeric(df["UC"], errors="coerce")
            neg = int((uc < 0).sum())
            non_int_ratio = float(((uc % 1) != 0).mean()) if uc.notna().any() else 0.0
            too_high = int((uc > 200).sum())
            if neg:
                issues.append(ValidationIssue("uc_negative", {"rows": neg}))
            if non_int_ratio > 0.05:
                issues.append(ValidationIssue("uc_non_integer_ratio", {"ratio": non_int_ratio}))
            if too_high:
                issues.append(ValidationIssue("uc_suspiciously_high", {"rows": too_high}))

        return issues

    def check_minimum_duration_requirement(
        self,
        df: pd.DataFrame,
        min_required_min: int = 10,
        duration_column: Optional[str] = None,
    ) -> List[ValidationIssue]:
        # Alias to time consistency; separated for report readability
        return self.check_time_consistency(df, min_duration_min=min_required_min, duration_column=duration_column)

    # ---------- Report ----------
    def quality_report(
        self,
        df: pd.DataFrame,
        duration_column: Optional[str] = None,
        expected_sampling_hz: int = 4,
        min_duration_min: int = 10,
    ) -> pd.DataFrame:
        """Aggregate checks into a tidy report table with summary rows and guidance."""
        issues: List[ValidationIssue] = []
        issues += self.check_time_consistency(df, expected_sampling_hz, min_duration_min, duration_column)
        issues += self.check_physiological_plausibility(df)

        records = [{"type": it.type, **it.details} for it in issues]
        report = pd.DataFrame(records)

        # Validity estimation: rows not implicated by severe issues
        n = len(df)
        invalid_rows_estimate = 0
        # Count rows implicated by out-of-physiology baseline and percentage bounds as severe
        for t in ["lb_out_of_physiology", "percentage_out_of_bounds", "sensor_inconsistency_zero_var_with_events"]:
            if not report.empty and t in report["type"].values:
                subset = report[report["type"] == t]
                invalid_rows_estimate += int(subset["rows"].sum()) if "rows" in subset.columns else 0
        valid_pct = max(0.0, 100.0 * (1.0 - invalid_rows_estimate / n)) if n else 0.0

        summary_rows = [
            {"type": "summary_valid_percentage", "percent": valid_pct},
            {"type": "guideline_citation", "text": CTG_GUIDELINES_CITATION},
        ]

        # Recommendations based on common findings
        recs: List[str] = []
        if not report.empty and "uc_non_integer_ratio" in report["type"].values:
            recs.append("Round/validate UC to non-negative integers; verify acquisition settings")
        if not report.empty and "lb_outside_histogram_range" in report["type"].values:
            recs.append("Review histogram-derived features and baseline alignment")
        if not report.empty and "sensor_inconsistency_zero_var_with_events" in report["type"].values:
            recs.append("Investigate sensor placement/connection; zero variability with events is suspect")
        if not report.empty and "duration_unknown" in report["type"].values:
            recs.append("Capture recording duration explicitly to meet minimal analysis requirements")

        if recs:
            summary_rows.append({"type": "recommendations", "text": "; ".join(recs)})

        summary_df = pd.DataFrame(summary_rows)
        full_report = pd.concat([report, summary_df], ignore_index=True)
        logger.info("Generated CTG validation report: %d issues + summary", len(report))
        return full_report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # Example manual run (requires a DataFrame to validate)
    # from src.data.data_loader import DataLoader
    # df = DataLoader().clean_ctg_dataframe(DataLoader().load(sheet="Data"))
    # report = CTGValidator().quality_report(df)
    # report.to_csv("reports/validation_report.csv", index=False)


