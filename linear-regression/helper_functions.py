import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PowerTransformer
from typing import Tuple
import numpy as np
import pytest


def create_equatorial_distance_feature(df) -> pd.DataFrame:
    """
    Create a new feature 'distance_from_equator' based on the absolute value of 'Latitude' and bin the values into 10 bins.

    Args:
        df: pandas DataFrame - Input DataFrame containing 'Latitude' column.

    Returns:
        pandas DataFrame - DataFrame with the new feature 'distance_from_equator' and binned values in 'distance_from_equator_bin'.
    """

    # Create a new feature 'distance_from_equator' by taking the absolute value of 'Latitude'
    df['distance_from_equator'] = df['Latitude'].abs()

    # Bin the 'distance_from_equator' values into 10 bins and name it 'distance_from_equator_bin'
    # use the pd.cut() function to bin the values
    df['distance_from_equator_bin'] = pd.cut(df['distance_from_equator'], bins=10, labels=False)
    return df


def evaluate_model(y_true, y_pred):
    """
    Calculate Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R2) score between true and predicted values.

    Args:
        y_true: Array-like of shape (n_samples,) - Ground truth target values.
        y_pred: Array-like of shape (n_samples,) - Predicted target values.

    Returns:
        mse: float - Mean Squared Error.
        mae: float - Mean Absolute Error.
        r2: float - R-squared score.

    Examples:
        mse, mae, r2 = evaluate_model(y_true, y_pred)
    """
    # create the mse, mae, and r2 variables
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)


    # return the mse, mae, and r2 values
    return mse, mae, r2


def linear_regression_pipeline(
    X_train, X_test, y_train, y_test, target_transformer=None
):
    """
    Perform a linear regression pipeline including data splitting, model training, and evaluation.

    Args:
        X_train: pandas DataFrame - Contains training data features.
        X_test: pandas DataFrame - Contains test data features.
        y_train: pandas Series - Contains training data target.
        y_test: pandas Series - Contains test data target.
        target_transformer: PowerTransformer - Target transformer object to inverse transform the target variable.

    Returns:
        model: LinearRegression - Trained linear regression model.
        train_metrics: tuple - Mean Squared Error, Mean Absolute Error, and R-squared for training data.
        test_metrics: tuple - Mean Squared Error, Mean Absolute Error, and R-squared for testing data.
    """


    from sklearn.linear_model import LinearRegression

    # create and fit a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions on the X_train and X_test datasets
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)


    # inverse transform all 4 target variables (y_train, y_pred_train, y_test, y_pred_test)
    if target_transformer:
         y_train = target_transformer.inverse_transform(y_train.values.reshape(-1, 1)) # -1 reduce it from 2 dimentional to 1 dimentional
         y_pred_train = target_transformer.inverse_transform(y_pred_train.reshape(-1, 1))
         y_test = target_transformer.inverse_transform(y_test.values.reshape(-1, 1))
         y_pred_test = target_transformer.inverse_transform(y_pred_test.reshape(-1, 1))

    # evaluate the model using the evaluate_model function - for both train and test sets
    train_metrics = evaluate_model(y_train, y_pred_train)
    test_metrics = evaluate_model(y_test, y_pred_test)

    # return the model, train_metrics, and test_metrics
    return model, train_metrics, test_metrics


def scale_features(X_train, X_test, feature_cols) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scales the specified feature columns in the training and testing datasets using StandardScaler.

    Args:
        X_train: A pandas DataFrame containing the training dataset.
        X_test: A pandas DataFrame containing the testing dataset.
        feature_cols: A list of column names to be scaled.

    Returns:
        Scaled pandas DataFrames for X_train and X_test.

    Raises:
        No specific exceptions are raised.
    """

    from sklearn.preprocessing import StandardScaler

    # create a StandardScaler object and fit it to the training data with the specified feature columns
    rs = StandardScaler()
    rs.fit(X_train[feature_cols])

    # transform the feature columns in both the training and testing datasets
    X_train[feature_cols] = rs.transform(X_train[feature_cols])
    X_test[feature_cols] = rs.transform(X_test[feature_cols])
    

    # return the scaled training and testing datasets
    return X_train, X_test




# not for students to fill out
@pytest.mark.skip(reason="helper function - not for students to fill out")
def pretty_metrics(
    model,
    train_metrics: Tuple[float, float, float],
    test_metrics: Tuple[float, float, float],
):
    """
    Creates a DataFrame to store the metrics of both train and test sets for a given model.

    Args:
        model: The model for which metrics are being calculated.
        train_metrics: A tuple containing the metrics (mse, mae, r2) for the training set.
        test_metrics: A tuple containing the metrics (mse, mae, r2) for the testing set.

    Returns:
        A concatenated pandas DataFrame containing the metrics for both train and test sets.

    Raises:
        No specific exceptions are raised.
    """

    # create a dataframe to store the metrics of both train and test sets
    train_metrics_df = pd.DataFrame(
        {
            "model_name": model.__class__.__name__,
            "data": "train",
            "mse": train_metrics[0],
            "rmse": np.sqrt(train_metrics[0]),
            "mae": train_metrics[1],
            "r2": train_metrics[2],
        },
        index=["train"],
    )

    test_metrics_df = pd.DataFrame(
        {
            "model_name": model.__class__.__name__,
            "data": "test",
            "rmse": np.sqrt(test_metrics[0]),
            "mse": test_metrics[0],
            "mae": test_metrics[1],
            "r2": test_metrics[2],
        },
        index=["test"],
    )

    return pd.concat([train_metrics_df, test_metrics_df], axis=0).reset_index(drop=True)




def transform_target(
    y_train: pd.Series, y_test: pd.Series
) -> Tuple[pd.Series, pd.Series, PowerTransformer]:
    """
    Transforms the target variables using PowerTransformer based on the method chosen.

    Args:
        y_train: A pandas Series containing the training target variable.
        y_test: A pandas Series containing the testing target variable.

    Returns:
        Transformed pandas Series for y_train and y_test.
        The target transformer object.

    Raises:
        No specific exceptions are raised.

    Examples:
        y_train, y_test, target_transformer = transform_target(y_train, y_test)
    """

    # test if all the target values in y_train and y_test are positive
    # if positive, use the box-cox method, otherwise use the yeo-johnson method
    # Create the PowerTransformer object and fit it to the training target variable here
    if (y_train > 0).all() and (y_test > 0).all():
        transformer = PowerTransformer(method='box-cox')
    else:
        transformer = PowerTransformer(method='yeo-johnson')

    try:
        transformer.fit(y_train.values.reshape(-1, 1))
    except KeyboardInterrupt:
        print('KeyboardInterrupt Error')


    # Use the fitted transformer to transform the target variables in both the training and testing datasets
    # note you will have to reshape the target variables to a 2D array before transforming `values.reshape(-1, 1)`
    y_train = transformer.transform(y_train.values.reshape(-1, 1)).flatten()
    y_test = transformer.transform(y_test.values.reshape(-1, 1)).flatten() # Return a copy of the array collapsed into one dimension.

    # return the transformed target variables and the target transformer object
    return pd.Series(y_train), pd.Series(y_test), transformer




def backward_stepwise_train(X_train, y_train, feature_cols):
    """
    Perform backward stepwise selection to choose the best subset of
    features to use for linear regression based on RMSE. This function
    should also split the data into train and validation subsets and the RMSE should
    be calculated based on the validation data.

    Args:
        X_train: pandas DataFrame - Input DataFrame containing all training data.
        y_train: pandas Series - Series containing target values for training data.
        feature_cols: list[str] - Names of feature columns to use for starting (full) model.

    Returns:
        best_idx: int - Index of the best model based on the lowest RMSE.
        features_list: dict - Dictionary containing the list of features used for each model.
        rmse_list: list - List of RMSE values for each model.
    """

    from sklearn.model_selection import train_test_split


    # split the data into training and validation sets (xtrain, xval, ytrain, yval)
    # use test size 0.2 and random state 1
    xtrain, xval, ytrain, yval = train_test_split(X_train, y_train, test_size=0.2, random_state=1)



    # transform the target variable using the transform_target function
    ytrain, yval, target_transformer = transform_target(ytrain, yval)


     # Check if xtrain and ytrain have the same number of samples
    if xtrain.shape[0] != ytrain.shape[0]:
        print(f"Xtrain: {xtrain.shape}, ytrain: {ytrain.shape}")
        raise ValueError("xtrain and ytrain must have the same number of samples")
    
    # create an empty list to store the best models
    best_models = []
    
    # create an empty list to store the RMSE values.  Also store the first set of features in it
    remaining_features = feature_cols.copy()
    
    # create an empty list to store the RMSE values
    rmse_list = []
    
    # create an empty dictionary to store the features used for each model
    features_list = {}
    
    # create a variable k to store the number of features
    k = len(feature_cols)

    # Get full model
    model, _, _ = linear_regression_pipeline(
        xtrain, xval, ytrain, yval, target_transformer
    )

    # calculate the RMSE for the full model, make sure to inverse transform the target variable
    y_pred_val = model.predict(xval)
    y_pred_val_inv = target_transformer.inverse_transform(y_pred_val.reshape(-1,1))
    y_val_inv = target_transformer.inverse_transform(yval.values.reshape(-1,1))
    mse = mean_squared_error(y_val_inv, y_pred_val_inv)
    rmse = np.sqrt(mse)

    # append the full-featured model to the best_models list
    best_models.append(model)

    # save the features in the features_list
    features_list[0] = feature_cols.copy()

    # append the rmse to the rmse_list
    rmse_list.append(rmse)

    # Loop through the features
    for i in range(1, k):
        best_rmse = np.inf
        best_model = None

        # Loop through the remaining features
        for feat in remaining_features:
            features = remaining_features.copy()
            
            # remove the current feature from the features list
            features.remove(feat)
            
            # train a model using the remaining features
            model, _, _ = linear_regression_pipeline(
                xtrain[features], xval[features], ytrain, yval, target_transformer
            )
            # calculate the RMSE for the model, make sure to inverse transform the target variable
            y_pred_val = model.predict(xval[features])
            y_pred_val_inv = target_transformer.inverse_transform(y_pred_val.reshape(-1,1))
            y_val_inv = target_transformer.inverse_transform(yval.values.reshape(-1,1))
            mse = mean_squared_error(y_val_inv, y_pred_val_inv)
            rmse = np.sqrt(mse)

            # If the current rmse is lower than the best rmse, update the best rmse, worst feature, and best model
            if rmse < best_rmse:
                best_rmse = rmse
                worst_feature = feat
                best_model = model

        # Save best model in the list
        best_models.append(best_model)

        # Updating variables for next loop
        remaining_features.remove(worst_feature)

        # Save (append) the best_rmse in the rmse_list
        rmse_list.append(best_rmse)

        # Save the remaining features in the features_list
        features_list[i] = remaining_features.copy()

    # Find the index of the best model based on the lowest RMSE
    best_idx = np.argmin(rmse_list)

    # Return the index of the best model, the features used, and the RMSE values
    return best_idx, features_list, rmse_list




def build_lasso_model(X_train, y_train, alpha):
    """
    Build a LASSO regression model with the given regularization constant (alpha).
    This function should also split the data into train and validation subsets and
    the RMSE should be calculated based on the validation data.

    Args:
        X_train: pandas DataFrame - Input DataFrame containing all training data.
        y_train: pandas Series - Series containing target values for training data.
        alpha: int - regularization constant.

    Returns:
        feature_coefs: list[float] - Coefficients of features.
        best_rmse: float - RMSE for best model.
    """

    from sklearn.linear_model import Lasso
    from sklearn.model_selection import train_test_split

    # split the data into training and validation sets (xtrain, xval, ytrain, yval)
    # use test size 0.2 and random state 1
    xtrain, xval, ytrain, yval = train_test_split(
        X_train, y_train, test_size=0.2, random_state=1
    )

    # transform the target variable using the transform_target function
    ytrain, yval, target_transformer = transform_target(ytrain, yval)
    
    # Create Lasso object and fit model
    lasso = Lasso(alpha=alpha)
    model = lasso.fit(xtrain, ytrain)
    

    # Set number of observations (n) and number of non-zero parameters (p)
    n = xtrain.shape[0]
    p = np.count_nonzero(model.coef_)

    # Predict validation set and convert target back to original units
    # Hint: Call reshape(-1, 1) on predictions before inverse transforming
    preds = model.predict(xval).reshape(-1, 1)
    yval = target_transformer.inverse_transform(yval.values.reshape(-1, 1))
    preds = target_transformer.inverse_transform(preds)

    # Calculate RMSE based on number of non-zero coefficients
    sse = np.sum((preds - yval) ** 2)
    rmse = np.sqrt(sse / n)

    # Return model coefficients and RMSE
    return model.coef_, rmse
