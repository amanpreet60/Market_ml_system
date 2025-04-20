import sys
sys.path.append('/Users/amanpreetsingh/My Computer/VSCode/Market')


import logging
from src.model_evaluator import RegressionModelEvaluation
from zenml import Model, step, model
from sklearn.pipeline import Pipeline
import pandas as pd
from typing import Tuple
from zenml import Model, pipeline, step, model
from xgboost import XGBRegressor
import mlflow


@step(enable_cache=False)
def model_evaluator_step(
    trained_model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.DataFrame
) -> Tuple[dict, float]:
    
    # Ensure the inputs are of the correct type
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    if not isinstance(y_test, pd.DataFrame):
        raise TypeError("y_test must be a pandas DataFrame.")

    logging.info("Applying the same preprocessing to the test data.")

    evaluation_metrics = RegressionModelEvaluation().evaluate_model(trained_model,X_test=X_test,y_test=y_test)

    # Ensure that the evaluation metrics are returned as a dictionary
    if not isinstance(evaluation_metrics, dict):
        raise ValueError("Evaluation metrics must be returned as a dictionary.")
    rmse = evaluation_metrics.get("Root Mean Squared Error", None)

    return evaluation_metrics, rmse
