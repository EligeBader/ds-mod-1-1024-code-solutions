# TODO: 
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report

def grid_train_random_forest(X, y, params, n_folds=4, eval_metric='accuracy'):
    """
    Train Random Forest binary classifier using a grid of hyperparameters. Return
    the best model according to the specified metric.

    Args:
        X: Array-like of shape (n_samples,n_features) - Training feature data.
        y: Array-like of shape (n_samples,) - Training target data.
        params: Dictionary - Parameter grid on which to perform cross validation.
        n_folds: int - Number of folds to use for cross validation.
        eval_metric: str - Metric to use for evaluating model performance in cross validation.

    Returns:
        model: Best Random Forest model according to evaluation metric.

    Examples:
        model = grid_train_random_forest(X, y, params, 4, "accuracy")
    """

    # TODO: Implement this function
    rf = RandomForestClassifier()
    gs = GridSearchCV(rf, param_grid=params, cv=n_folds, scoring=eval_metric)
    gs.fit(X, y)
    rf_best = gs.best_estimator_

    return rf_best




def calc_roc_metrics(X, y, model):
    """
    Calculate False Positive Rate (FPR), True Positive Rate (TPR), and Area Under ROC Curve (AUC)
    for a given binary classification model and test data.

    Args:
        X: Array-like of shape (n_samples,n_features) - Test feature data.
        y: Array-like of shape (n_samples,) - Test target data.
        model: Scikit-learn style binary classification model.

    Returns:
        fpr: np.array[float] - False Positive Rates.
        tpr: np.array[float] - True Positive Rates.
        auc: float - Area Under ROC Curve.

    Examples:
        fpr, tpr, auc = calc_roc_metrics(X, y, model)
    """

    # TODO: Implement this function
    y = np.where(y == 'No', 0, 1)
    yproba = model.predict_proba(X)
    fpr, tpr, auc = roc_curve(y, yproba[:,1])
    auc_val = roc_auc_score(y, yproba[:,1])

    return fpr, tpr, auc_val



def train_xgboost(X_train, y_train, X_test, y_test, params, n_round):
    """
    Train an XGBoost model with the given parameters and train/test data.

    Args:
        X_train: Array-like of shape (n_train_samples,n_features) - Train feature data.
        y_train: Array-like of shape (n_train_samples,) - Train target data.
        X_test: Array-like of shape (n_test_samples,n_features) - Test feature data.
        y_test: Array-like of shape (n_test_samples,) - Test target data.
        params: Dictionary - Parameters to pass into XGBoost trainer.
        n_round: int - Number of rounds of training.

    Returns:
        model: Trained XGBoost model.

    Examples:
        model = calc_roc_metrics(X_train, y_train, X_test, y_test, params)
    """

    # TODO: Implement this function
    
    y_train = np.where(y_train == 'No', 0, 1)
    y_test = np.where(y_test == 'No', 0, 1)

    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    
    model = xgb.train(params, dtrain, n_round, evals=[(dtest, 'test')])
    
    return model

