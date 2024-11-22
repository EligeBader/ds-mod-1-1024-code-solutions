import pytest
import pandas as pd
import numpy as np

from datetime_engineering import convert_datetime, create_month, aggregate_months

df_t = pd.DataFrame({"Date": ["Jan 1, 2020", "Mar 3, 2024"], "B": [3, 4]})
date = pd.to_datetime(df_t.Date)
date_df = df_t.copy()
date_df["Date"] = date

month = date.apply(lambda x: x.month)
month_df = date_df.copy()
month_df["Month"] = month

group_df = month_df.groupby("Month").agg("mean")

@pytest.mark.parametrize(
    "df, feature, expected_df",
    [
        # Happy path test cases
        pytest.param(
            df_t,
            "Date",
            date_df,
            id="happy_path",
        )
    ],
)
def test_convert_datetime(df, feature, expected_df):
    # Act
    result_df = convert_datetime(df, feature)

    # Assert
    pd.testing.assert_frame_equal(result_df, expected_df)

@pytest.mark.parametrize(
    "df, feature, name, expected_df",
    [
        # Happy path test cases
        pytest.param(
            date_df,
            "Date",
            "Month",
            month_df,
            id="happy_path",
        )
    ],
)
def test_create_month(df, feature, name, expected_df):
    # Act
    result_df = create_month(df, feature, name)

    # Assert
    pd.testing.assert_frame_equal(result_df, expected_df)

@pytest.mark.parametrize(
    "df, group, agg_func, expected_df",
    [
        # Happy path test cases
        pytest.param(
            month_df,
            "Month",
            "mean",
            group_df,
            id="happy_path",
        )
    ],
)
def test_aggregate_months(df, group, agg_func, expected_df):
    # Act
    result_df = aggregate_months(df, group, agg_func)

    # Assert
    pd.testing.assert_frame_equal(result_df, expected_df)