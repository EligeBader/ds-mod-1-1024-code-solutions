import pytest
import pandas as pd
from target_encoding import discretize_feature, target_encode


@pytest.mark.parametrize(
    "df, feature, expected_df",
    [
        # Happy path
        (pd.DataFrame({"A": [1.1, 2.5, 3.9]}), "A", pd.DataFrame({"A": [1, 2, 4]})),
        # Edge cases
        (pd.DataFrame({"A": [0.4, 0.5, 0.6]}), "A", pd.DataFrame({"A": [0, 0, 1]})),
        (
            pd.DataFrame({"A": [-1.1, -2.5, -3.9]}),
            "A",
            pd.DataFrame({"A": [-1, -2, -4]}),
        ),
    ],
    ids=["happy_path", "edge_case_0.5", "negative_values"],
)
def test_discretize_feature(df, feature, expected_df):
    # Act
    result_df = discretize_feature(df, feature)

    # Assert
    pd.testing.assert_frame_equal(
        result_df, expected_df, check_column_type=False, check_dtype=False
    )


@pytest.mark.parametrize(
    "df, features_to_encode, target_col, expected_df",
    [
        # Happy path
        (
            pd.DataFrame({"A": ["a", "b", "a"], "B": [1, 2, 3]}),
            ["A"],
            "B",
            pd.DataFrame({"A": [2.0, 2.0, 2.0], "B": [1, 2, 3]}),
        ),
        # Edge cases
        (
            pd.DataFrame({"A": ["a", "a", "a"], "B": [1, 1, 1]}),
            ["A"],
            "B",
            pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [1, 1, 1]}),
        ),
        # # Error cases
        # (
        #     pd.DataFrame({"A": ["a", "b", "c"], "B": [1, 2, 3]}),
        #     ["A"],
        #     "B",
        #     pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [1, 2, 3]}),
        # ),
    ],
    ids=["happy_path", "single_category"],
)
def test_target_encode(df, features_to_encode, target_col, expected_df):
    # Act
    result_df = target_encode(df, features_to_encode, target_col)

    # Assert
    pd.testing.assert_frame_equal(result_df, expected_df)
