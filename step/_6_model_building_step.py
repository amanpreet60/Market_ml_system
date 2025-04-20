import logging
from typing import Annotated
import sys
sys.path.append('/Users/amanpreetsingh/My Computer/VSCode/Market')

import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from zenml import ArtifactConfig, step
from zenml.client import Client
from src.model_building import XGBoost
from xgboost import XGBRegressor

# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker
mlflow.set_tracking_uri(Client().active_stack.experiment_tracker.get_tracking_uri())
from zenml import Model

model = Model(
    name="prices_predictor",
    version=None,
    license="Apache 2.0",
    description="Price prediction model for houses.",
)


@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(
    X_train: pd.DataFrame, y_train: pd.DataFrame
) -> XGBRegressor:
    
    # Ensure the inputs are of the correct type
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.DataFrame):
        raise TypeError("y_train must be a pandas DataFrame.")

    # Start an MLflow run to log the model training process
    if not mlflow.active_run():
        mlflow.start_run()  # Start a new MLflow run if there isn't one active

    try:
        # Enable autologging to automatically capture model metrics, parameters, and artifacts
        mlflow.xgboost.autolog()
        mlflow.sklearn.autolog()

        logging.info("Building and training the XgBoost model.")
        best_xgb = XGBoost().build_and_train_model(X_train=X_train,y_train=y_train)
        logging.info("Model training completed.")

        expected_columns = list(X_train.columns)
        logging.info(f"Model expects the following columns: {expected_columns}")
    
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

    finally:
        # End the MLflow run
        mlflow.end_run()

    return best_xgb

'''df = pd.read_csv('/Users/amanpreetsingh/My Computer/VSCode/Market/house_price_data/train.csv')
from sklearn.model_selection import train_test_split
X = df.drop(columns=['SalePrice'])  # Features (independent variables)
y = df['SalePrice']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = pd.DataFrame(y_train, columns=['SalePrice'])
model_building_step(X_train=X_train, y_train=y_train)'''
