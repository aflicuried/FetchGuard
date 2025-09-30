from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


EPS = 1e-6


@dataclass
class FeatureInfo:
    name: str
    description: str
    formula: str
    group: str


class MedicalScaler:
    """Create clinically interpretable scaled features without overwriting raw columns.

    Scaling rules are group-specific and aim to retain clinical meaning, e.g. scaling FHR-based
    values to [0, 1] within plausible physiological bounds rather than z-scoring.
    """

    def __init__(self) -> None:
        # Clinical bounds (conservative) for scaling
        self.bounds: Dict[str, Tuple[float, float]] = {
            "LB": (50.0, 240.0),
            "Min": (0.0, 240.0),
            "Max": (0.0, 240.0),
            "Mode": (0.0, 240.0),
            "Mean": (0.0, 240.0),
            "Median": (0.0, 240.0),
            "MSTV": (0.0, 10.0),
            "MLTV": (0.0, 50.0),
            "ASTV": (0.0, 100.0),
            "ALTV": (0.0, 100.0),
        }

    def scale(self, df: pd.DataFrame, columns: Optional[List[str]] = None, suffix: str = "_scaled") -> pd.DataFrame:
        df = df.copy()
        cols = columns or [c for c in df.columns if c in self.bounds]
        for col in cols:
            if col not in df:
                continue
            lo, hi = self.bounds[col]
            s = pd.to_numeric(df[col], errors="coerce")
            scaled = (s - lo) / max(hi - lo, EPS)
            df[f"{col}{suffix}"] = scaled.clip(0.0, 1.0)
        logger.info("Applied medical scaling to %d columns", len(cols))
        return df


class FeatureEngineer:
    """Domain-specific feature engineering for CTG aggregated tabular dataset.

    Note: The UCI CTG dataset provides per-record aggregates, not raw time series. Time-based
    features here are proxies (e.g., using 'Tendency'). If true time-series columns are provided,
    supply them via optional parameters and the engineer will compute rolling stats accordingly.
    """

    def __init__(self, target_column: str = "NSP", duration_column: Optional[str] = None) -> None:
        self.target_column = target_column
        self.duration_column = duration_column
        self.feature_infos: List[FeatureInfo] = []
        self.scaler = MedicalScaler()

        # Canonical CTG column names mapping (case-insensitive)
        self.alias_map: Dict[str, str] = {
            "lb": "LB", "ac": "AC", "fm": "FM", "uc": "UC",
            "dl": "DL", "ds": "DS", "dp": "DP",
            "astv": "ASTV", "mstv": "MSTV", "altv": "ALTV", "mltv": "MLTV",
            "width": "Width", "min": "Min", "max": "Max", "nmax": "Nmax", "nzeros": "Nzeros",
            "mode": "Mode", "mean": "Mean", "median": "Median", "variance": "Variance", "tendency": "Tendency",
            "class": "CLASS", "nsp": "NSP",
        }

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        rename = {}
        for c in df.columns:
            key = str(c).strip().lower()
            if key in self.alias_map:
                rename[c] = self.alias_map[key]
        if rename:
            df = df.rename(columns=rename)
        return df

    def _repair_header_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """If the actual headers are in the first/second data row, promote them to columns.

        This does NOT change upstream cleaning; it only fixes the frame we received here.
        """
        expected = {
            "LB", "AC", "FM", "UC", "DL", "DS", "DP",
            "ASTV", "MSTV", "ALTV", "MLTV",
            "Width", "Min", "Max", "Nmax", "Nzeros",
            "Mode", "Mean", "Median", "Variance", "Tendency",
            "CLASS", "NSP",
        }
        for row_idx in [0, 1]:
            if row_idx >= len(df):
                break
            values = [str(v).strip() for v in df.iloc[row_idx].values]
            match = sum(1 for v in values if v in expected)
            if match >= 10:
                # Promote this row to header
                df = df.copy()
                df.columns = values
                df = df.iloc[row_idx + 1 :].reset_index(drop=True)
                logger.info("Promoted row %d to header for feature engineering", row_idx)
                break
        # Normalize any promoted headers
        df = self._normalize_columns(df)
        return df

    # ---------- Core domain features ----------
    def build_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Variability indices
        if set(["MSTV", "MLTV"]).issubset(df.columns):
            df["stv_ltv_ratio"] = (pd.to_numeric(df["MSTV"], errors="coerce") + EPS) / (
                pd.to_numeric(df["MLTV"], errors="coerce") + EPS
            )
            self._track("stv_ltv_ratio", "Short-to-Long-term variability ratio", "MSTV/MLTV", "variability")
            df["variability_spread"] = pd.to_numeric(df["MLTV"], errors="coerce") - pd.to_numeric(df["MSTV"], errors="coerce")
            self._track("variability_spread", "Difference between LTV and STV", "MLTV - MSTV", "variability")

        if set(["ASTV", "ALTV"]).issubset(df.columns):
            df["abnormal_stv_ltv_ratio"] = (pd.to_numeric(df["ASTV"], errors="coerce") + EPS) / (
                pd.to_numeric(df["ALTV"], errors="coerce") + EPS
            )
            self._track("abnormal_stv_ltv_ratio", "Abnormal STV to LTV percent ratio", "ASTV/ALTV", "variability")

        # Deceleration severity score (weighted counts)
        for c in ["DL", "DS", "DP"]:
            if c not in df.columns:
                df[c] = np.nan
        decel = pd.DataFrame({
            "DL": pd.to_numeric(df["DL"], errors="coerce").fillna(0),
            "DS": pd.to_numeric(df["DS"], errors="coerce").fillna(0),
            "DP": pd.to_numeric(df["DP"], errors="coerce").fillna(0),
        })
        df["decel_severity"] = (1.0 * decel["DL"] + 2.0 * decel["DS"] + 3.0 * decel["DP"]) / (
            decel.sum(axis=1) + EPS
        )
        self._track(
            "decel_severity",
            "Weighted deceleration severity score",
            "(1*DL + 2*DS + 3*DP)/(DL+DS+DP)",
            "deceleration",
        )

        # Acceleration pattern
        if "AC" in df.columns:
            ac = pd.to_numeric(df["AC"], errors="coerce")
            uc = pd.to_numeric(df["UC"], errors="coerce") if "UC" in df.columns else np.nan
            df["accel_per_uc"] = ac / (uc + 1.0)
            self._track("accel_per_uc", "Accelerations per uterine contraction", "AC/(UC+1)", "acceleration")
            mstv = pd.to_numeric(df["MSTV"], errors="coerce") if "MSTV" in df.columns else np.nan
            df["accel_per_stv"] = ac / (mstv + EPS)
            self._track("accel_per_stv", "Accelerations relative to STV", "AC/MSTV", "acceleration")
            if self.duration_column and self.duration_column in df.columns:
                duration_min = pd.to_numeric(df[self.duration_column], errors="coerce")
                df["accel_per_10min"] = ac / (duration_min / 10.0 + EPS)
                self._track("accel_per_10min", "Accelerations per 10 minutes", "AC/(duration_min/10)", "acceleration")

        # Baseline stability
        if set(["LB", "MLTV"]).issubset(df.columns):
            ltv = pd.to_numeric(df["MLTV"], errors="coerce")
            df["baseline_stability"] = 1.0 / (1.0 + ltv)
            self._track("baseline_stability", "Inverse long-term variability as stability", "1/(1+MLTV)", "baseline")
        if set(["LB", "Mode"]).issubset(df.columns):
            lb = pd.to_numeric(df["LB"], errors="coerce")
            mode = pd.to_numeric(df["Mode"], errors="coerce")
            df["baseline_mode_gap"] = (lb - mode).abs()
            self._track("baseline_mode_gap", "Absolute gap between baseline and mode", "|LB-Mode|", "baseline")

        return df

    # ---------- Time-based proxies and trends ----------
    def build_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Use existing Tendency as a trend proxy
        if "Tendency" in df.columns:
            tend = pd.to_numeric(df["Tendency"], errors="coerce")
            df["trend_strength"] = tend.abs()
            df["trend_direction"] = np.sign(tend).astype(float)
            self._track("trend_strength", "Absolute histogram tendency as trend strength", "|Tendency|", "trend")
            self._track("trend_direction", "Direction of trend from histogram tendency", "sign(Tendency)", "trend")

        # If true time-based columns are available, compute rolling proxies (optional extension)
        # Placeholder: log-only, to avoid misleading engineered features
        return df

    # ---------- Interaction features ----------
    def build_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "LB" in df.columns and "UC" in df.columns:
            df["fhr_uc_interaction"] = pd.to_numeric(df["LB"], errors="coerce") * pd.to_numeric(df["UC"], errors="coerce")
            self._track("fhr_uc_interaction", "Interaction of baseline FHR and uterine contractions", "LB*UC", "interaction")
        if "AC" in df.columns and "LB" in df.columns:
            df["ac_baseline_ratio"] = pd.to_numeric(df["AC"], errors="coerce") / (pd.to_numeric(df["LB"], errors="coerce") + EPS)
            self._track("ac_baseline_ratio", "Accelerations relative to baseline FHR", "AC/LB", "interaction")
        if set(["DL", "DS", "DP", "MSTV", "MLTV"]).issubset(df.columns):
            decel_total = (
                pd.to_numeric(df["DL"], errors="coerce").fillna(0)
                + pd.to_numeric(df["DS"], errors="coerce").fillna(0)
                + pd.to_numeric(df["DP"], errors="coerce").fillna(0)
            )
            var_total = (pd.to_numeric(df["MSTV"], errors="coerce").fillna(0) + pd.to_numeric(df["MLTV"], errors="coerce").fillna(0))
            df["decel_variability_ratio"] = decel_total / (var_total + EPS)
            self._track("decel_variability_ratio", "Decelerations relative to variability", "(DL+DS+DP)/(MSTV+MLTV)", "interaction")
        return df

    # ---------- Scaling ----------
    def add_scaled_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Create scaled counterparts for major clinical features
        to_scale = [c for c in ["LB", "Min", "Max", "Mode", "Mean", "Median", "MSTV", "MLTV", "ASTV", "ALTV"] if c in df.columns]
        return self.scaler.scale(df, columns=to_scale, suffix="_scaled")

    # ---------- Public pipeline ----------
    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Build engineered features and return (df_with_features, feature_metadata)."""
        logger.info("Starting feature engineering on shape=%s", df.shape)
        out = df.copy()
        # Attempt to repair misplaced headers (only local to FE stage)
        out = self._repair_header_rows(out)
        out = self._normalize_columns(out)
        # Coerce canonical numeric columns and drop rows fully empty on features
        numeric_cols = [
            c for c in [
                "LB", "AC", "FM", "UC", "DL", "DS", "DP",
                "ASTV", "MSTV", "ALTV", "MLTV",
                "Width", "Min", "Max", "Nmax", "Nzeros",
                "Mode", "Mean", "Median", "Variance", "Tendency",
            ] if c in out.columns
        ]
        for col in numeric_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        feature_subset = [c for c in out.columns if c not in {self.target_column, "CLASS"}]
        if feature_subset:
            out = out.dropna(axis=0, how="all", subset=feature_subset)
        out = self.build_domain_features(out)
        out = self.build_time_features(out)
        out = self.build_interactions(out)
        out = self.add_scaled_features(out)
        meta = self.feature_metadata()
        logger.info("Completed feature engineering: added %d features", len(meta))
        return out, meta

    # ---------- Metadata ----------
    def _track(self, name: str, description: str, formula: str, group: str) -> None:
        self.feature_infos.append(FeatureInfo(name=name, description=description, formula=formula, group=group))

    def feature_metadata(self) -> pd.DataFrame:
        if not self.feature_infos:
            return pd.DataFrame(columns=["name", "description", "formula", "group"])
        return pd.DataFrame([{ "name": fi.name, "description": fi.description, "formula": fi.formula, "group": fi.group } for fi in self.feature_infos])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # Example manual run
    # from src.data.data_loader import DataLoader
    # fe = FeatureEngineer()
    # df = DataLoader().clean_ctg_dataframe(DataLoader().load(sheet="Data"))
    # out, meta = fe.transform(df)
    # out.to_csv("data/processed/ctg_with_features.csv", index=False)
    # meta.to_csv("reports/feature_metadata.csv", index=False)


