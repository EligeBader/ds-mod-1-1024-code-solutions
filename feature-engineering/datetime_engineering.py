import pandas as pd
from typing import Callable

def convert_datetime(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Convert the specified feature to a datetime object and return modified dataframe.

    Args:
        df (pd.DataFrame): The input DataFrame containing the feature to be converted.
        feature (str): The name of the feature to be converted.

    Returns:
        pd.DataFrame: The DataFrame with the specified feature converted to datetime.
    """

    # TODO: Implement this function
    df[feature] = pd.to_datetime(df[feature])
    return df



def create_month(df: pd.DataFrame, feature: str, name: str) -> pd.DataFrame:
    """
    Convert a "month" feature from the specified datetime feature and return modified dataframe.

    Args:
        df (pd.DataFrame): The input DataFrame containing the datetime feature.
        feature (str): The name of the datetime feature.
        name (str): The name of new "month" feature to be created.

    Returns:
        pd.DataFrame: The DataFrame including the "month" feature.
    """

    # TODO: Implement this function
    df['Month'] = df[feature].dt.month
    return df


def aggregate_months(df: pd.DataFrame, group: str, agg_func: str) -> pd.DataFrame:
    """
    Perform aggregation function on dataframe based on grouping feature.

    Args:
        df (pd.DataFrame): The input DataFrame containing features to be aggregated.
        group (str): The name of the feature by which to group.
        agg_func (str): The name of the aggregating function.

    Returns:
        pd.DataFrame: The aggregated dataframe.
    """
   

    # TODO: Implement this function
    return df.groupby(group).agg(agg_func)
