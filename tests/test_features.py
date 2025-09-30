import pandas as pd
from src.features.build_features import basic_feature_set


def test_basic_feature_set_requires_target():
    df = pd.DataFrame({"a": [1, 2]})
    try:
        basic_feature_set(df)
        assert False, "Expected KeyError"
    except KeyError:
        assert True


