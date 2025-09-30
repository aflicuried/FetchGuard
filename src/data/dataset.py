from pathlib import Path
import pandas as pd


def load_ctg_raw(raw_path: Path) -> pd.DataFrame:
    """Load CTG dataset from Excel/CSV into a DataFrame.

    Accepts .xls/.xlsx or .csv files. Returns a DataFrame without changes.
    """
    if raw_path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(raw_path)
    if raw_path.suffix.lower() == ".csv":
        return pd.read_csv(raw_path)
    raise ValueError(f"Unsupported file format: {raw_path.suffix}")


