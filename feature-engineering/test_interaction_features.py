import pytest
import pandas as pd

from interaction_features import create_interaction_features


@pytest.mark.parametrize(
    "df, feature_pairs, expected_df",
    [
        # Happy path test cases
        pytest.param(
            pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
            [("A", "B")],
            pd.DataFrame({"A": [1, 2], "B": [3, 4], "A_x_B": [3, 8]}),
            id="happy_path_single_pair",
        ),
        pytest.param(
            pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]}),
            [("A", "B"), ("B", "C")],
            pd.DataFrame(
                {
                    "A": [1, 2],
                    "B": [3, 4],
                    "C": [5, 6],
                    "A_x_B": [3, 8],
                    "B_x_C": [15, 24],
                }
            ),
            id="happy_path_multiple_pairs",
        ),
        # Edge cases
        pytest.param(
            pd.DataFrame({"A": [0, 0], "B": [0, 0]}),
            [("A", "B")],
            pd.DataFrame({"A": [0, 0], "B": [0, 0], "A_x_B": [0, 0]}),
            id="edge_case_zeros",
        ),
        pytest.param(
            pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
            [],
            pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
            id="edge_case_empty_feature_pairs",
        ),
        # Error cases
        pytest.param(
            pd.DataFrame({"A": [1, 2]}),
            [("A", "B")],
            pd.DataFrame({"A": [1, 2]}),
            id="error_case_missing_feature",
            marks=pytest.mark.xfail(raises=KeyError),
        ),
    ],
)
def test_create_interaction_features(df, feature_pairs, expected_df):
    # Act
    result_df = create_interaction_features(df, feature_pairs)

    # Assert
    pd.testing.assert_frame_equal(result_df, expected_df)
