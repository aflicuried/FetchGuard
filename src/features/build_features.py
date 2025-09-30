from typing import Tuple
import pandas as pd


def basic_feature_set(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return a naive feature matrix X and target y from raw CTG DataFrame.

    This is a placeholder that expects a column named 'NSP' as target with
    values {1: Normal, 2: Suspect, 3: Pathologic} commonly used in CTG datasets.
    It drops the target and returns remaining columns as features.
    """
    if "NSP" not in df.columns:
        raise KeyError("Expected target column 'NSP' in dataset")
    y = df["NSP"].astype(int)
    X = df.drop(columns=["NSP"])
    return X, y


