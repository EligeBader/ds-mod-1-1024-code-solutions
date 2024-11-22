import pandas as pd


# fill out the following function
# the output should be a dataframe with the original features and the interaction features concatenated
# the name of each interaction feature should be the name of the original features joined by '_x_'
# example: if the original features are 'A' and 'B', the interaction feature should be 'A_x_B'


def create_interaction_features(
    df: pd.DataFrame, feature_pairs: list[tuple[str, str]]
) -> pd.DataFrame:
    """
    Creates interaction features by multiplying pairs of features in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the features.
        feature_pairs (list[tuple[str, str]]): List of tuples specifying pairs of features to multiply.

    Returns:
        pd.DataFrame: The DataFrame with added interaction features.

    Examples:
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        feature_pairs = [('A', 'B')]
        create_interaction_features(df, feature_pairs)  # Output: DataFrame with 'A_x_B' column
    """

    # TODO: Implement this function

    for pair in feature_pairs:
        feature1, feature2 = pair
        interaction_name = f"{feature1}_x_{feature2}"
        df[interaction_name] = df[feature1] * df[feature2]
    return df
