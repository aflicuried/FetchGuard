from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.io import arff  # type: ignore
    _HAS_ARFF = True
except Exception:  # pragma: no cover - optional dependency
    arff = None
    _HAS_ARFF = False


logger = logging.getLogger(__name__)


CTG_LABEL_MAP = {1: "Normal", 2: "Suspect", 3: "Pathologic"}


@dataclass
class DataStats:
    shape: Tuple[int, int]
    dtypes: Dict[str, str]
    missing: Dict[str, int]


class DataLoader:
    """Utility to load and inspect CTG datasets from data/raw.

    Supports .arff and .csv. Validates target column NSP and returns stats.
    """

    def __init__(self, raw_dir: Path = Path("data/raw")) -> None:
        self.raw_dir = raw_dir
        if not self.raw_dir.exists():
            logger.warning("Raw data directory does not exist: %s", self.raw_dir)
        # default sheet hint for CTG Excel files
        self.default_excel_sheet = "Data"

    # --------- public API ---------
    def load(self, filename: Optional[str] = None, sheet: Optional[str] = None) -> pd.DataFrame:
        """Load the dataset from raw directory.

        - If filename is provided, tries that file.
        - Otherwise prefers CTG.arff, then CTG.csv, then CTG.xls/.xlsx.
        Raises FileNotFoundError if no valid file found.
        """
        if filename is not None:
            candidate = self.raw_dir / filename
            logger.info("Attempting to load dataset: %s", candidate)
            return self._load_file(candidate, sheet=sheet)

        # search common names
        candidates = [
            self.raw_dir / "CTG.arff",
            self.raw_dir / "ctg.arff",
            self.raw_dir / "CTG.csv",
            self.raw_dir / "ctg.csv",
            self.raw_dir / "CTG.xls",
            self.raw_dir / "CTG.xlsx",
        ]
        for path in candidates:
            if path.exists():
                logger.info("Found dataset: %s", path)
                return self._load_file(path, sheet=sheet)

        # fallback: first ARFF/CSV/XLS(X) in directory
        for ext in ("*.arff", "*.csv", "*.xls", "*.xlsx"):
            found = list(self.raw_dir.glob(ext))
            if found:
                logger.info("Using first dataset found: %s", found[0])
                return self._load_file(found[0], sheet=sheet)

        raise FileNotFoundError(
            f"No dataset found in {self.raw_dir}. Expected .arff or .csv (.xls/.xlsx also supported)."
        )

    def get_stats(self, df: pd.DataFrame) -> DataStats:
        """Return basic stats: shape, dtypes, and missing counts."""
        shape = (df.shape[0], df.shape[1])
        dtypes = {c: str(t) for c, t in df.dtypes.items()}
        missing = df.isna().sum().to_dict()
        logger.debug("Data shape: %s, missing totals: %s", shape, sum(missing.values()))
        return DataStats(shape=shape, dtypes=dtypes, missing=missing)

    def split_feature_groups(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split features into FHR (fetal) and UC (uterine contraction) groups.

        Heuristic based on typical UCI CTG column names: FHR-related often include
        'LB', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'DL', 'DS', 'DP', 'DR', 'Width', 'Min', 'Max', etc.
        UC-related often include 'UC', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median' for toco.
        This function uses simple name matching; adjust to your schema as needed.
        """
        columns = list(df.columns)
        lower_map = {c: c.lower() for c in columns}

        fhr_like = {c for c in columns if any(k in lower_map[c] for k in [
            "fhr", "lb", "astv", "mstv", "altv", "mltv", "width", "min", "max", "dl", "ds", "dp", "dr"
        ])}
        uc_like = {c for c in columns if any(k in lower_map[c] for k in [
            "uc", "toco", "nzeros", "nmax", "mode", "mean", "median"
        ])}

        # ensure NSP and non-feature columns are excluded
        feature_cols = [c for c in columns if c != "NSP"]
        fhr_cols = [c for c in feature_cols if c in fhr_like]
        uc_cols = [c for c in feature_cols if c in uc_like and c not in fhr_like]

        if not fhr_cols:
            logger.warning("No FHR-like columns detected; verify column names.")
        if not uc_cols:
            logger.warning("No UC-like columns detected; verify column names.")

        return df[fhr_cols].copy(), df[uc_cols].copy()

    def get_target(self, df: pd.DataFrame, target_col: str = "NSP") -> pd.Series:
        """Extract NSP target (1=Normal, 2=Suspect, 3=Pathologic) with validation."""
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in dataset")
        y = df[target_col]
        if y.isna().any():
            raise ValueError("Target column contains missing values")
        unique = set(pd.unique(y))
        allowed = {1, 2, 3}
        if not unique.issubset(allowed):
            raise ValueError(f"Target contains values outside {allowed}: {sorted(unique)}")
        return y.astype(int)

    # --------- internal helpers ---------
    def _load_file(self, path: Path, sheet: Optional[str] = None) -> pd.DataFrame:
        if not path.exists():
            logger.error("Data file not found: %s", path)
            raise FileNotFoundError(str(path))

        suffix = path.suffix.lower()
        try:
            if suffix == ".csv":
                df = pd.read_csv(path)
            elif suffix == ".arff":
                if not _HAS_ARFF:
                    raise RuntimeError("ARFF support requires scipy; please install it.")
                data, meta = arff.loadarff(path)
                df = pd.DataFrame(data)
                # Convert bytes columns to str for categorical data
                for col in df.select_dtypes(["object"]).columns:
                    if df[col].map(lambda v: isinstance(v, (bytes, bytearray))).any():
                        df[col] = df[col].str.decode("utf-8")
                # Convert possible numpy record dtypes to float where applicable
                for col in df.columns:
                    if df[col].dtype.kind in {"S", "U", "O"}:
                        continue
                    # Cast numeric-likes to float
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except Exception:  # keep original
                        pass
            elif suffix in {".xls", ".xlsx"}:
                # Excel: allow explicit sheet override; otherwise select best sheet.
                # Revert to default header handling (use Excel's first header row).
                sheets = pd.read_excel(path, sheet_name=None)
                if not sheets:
                    raise ValueError(f"No sheets found in Excel file: {path}")

                def normalize(name: str) -> str:
                    return name.strip().lower()

                preferred_names = {"data", "raw data", "raw_data", "raw"}
                chosen_name = None

                # If a specific sheet name is provided and exists, use it directly
                if sheet is not None:
                    for name in sheets.keys():
                        if normalize(name) == normalize(sheet):
                            chosen_name = name
                            break

                # 1) Prefer by name
                if chosen_name is None:
                    for name in sheets.keys():
                        if normalize(name) in preferred_names:
                            chosen_name = name
                            break

                # 2) Prefer sheet containing NSP
                if chosen_name is None:
                    for name, sdf in sheets.items():
                        if "NSP" in sdf.columns:
                            chosen_name = name
                            break

                # 3) Fallback to the "largest" plausible sheet
                if chosen_name is None:
                    def sheet_score(sdf: pd.DataFrame) -> tuple[int, int]:
                        numeric_cols = sdf.select_dtypes(include=[np.number]).shape[1]
                        return (numeric_cols, sdf.shape[0])

                    chosen_name = max(sheets.items(), key=lambda kv: sheet_score(kv[1]))[0]

                # If default sheet is available, prefer it
                if self.default_excel_sheet and any(
                    normalize(n) == normalize(self.default_excel_sheet) for n in sheets.keys()
                ) and sheet is None:
                    chosen_name = next(n for n in sheets.keys() if normalize(n) == normalize(self.default_excel_sheet))
                df = sheets[chosen_name]
                logger.info("Selected Excel sheet '%s' from %s (shape=%s)", chosen_name, path.name, df.shape)
            else:
                raise ValueError(f"Unsupported file type: {suffix}")
        except Exception as exc:
            logger.exception("Failed loading data from %s", path)
            raise

        if df.empty:
            raise ValueError(f"Loaded empty dataset from {path}")

        logger.info("Loaded data: %s rows x %s cols from %s", df.shape[0], df.shape[1], path.name)
        return df

    # --------- CTG-specific cleaning helpers ---------
    @staticmethod
    def clean_ctg_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Original minimal CTG cleaning:
        - Drop separator/empty columns (Unnamed: x) and all-empty columns
        - Strip column names
        - Coerce numeric features where possible (keep CLASS/NSP as is, NSP -> Int)
        - Drop rows with all feature columns NaN
        """
        # Drop unnamed/separator columns and fully empty columns
        df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
        df = df.dropna(axis=1, how="all")

        # Normalize headers
        df.columns = df.columns.astype(str).str.strip()

        # Coerce numerics where possible (except CLASS/NSP)
        for col in df.columns:
            if col in {"CLASS", "NSP"}:
                continue
            df[col] = pd.to_numeric(df[col], errors="ignore")

        # Ensure NSP is numeric integer if present
        if "NSP" in df.columns:
            df["NSP"] = pd.to_numeric(df["NSP"], errors="coerce").astype("Int64")

        # Drop rows where all feature columns are NaN after coercion
        feature_cols = [c for c in df.columns if c not in {"CLASS", "NSP"}]
        if feature_cols:
            df = df.dropna(axis=0, how="all", subset=feature_cols)

        return df


