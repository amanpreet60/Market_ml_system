import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Model Building Strategy
class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> RegressorMixin:
        pass


# Concrete Strategy for XGBoost with Hyperparameter Tuning using RandomizedSearchCV
class XGBoost(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        # Ensure the inputs are of the correct type
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.DataFrame):
            raise TypeError("y_train must be a pandas DataFrame.")

        # Convert y_train from DataFrame to Series (1D array) for compatibility with XGBRegressor
        y_train = y_train.iloc[:, 0]  # Extract the first column if y_train is a DataFrame
        logging.info("Initializing XGBoost model with hyperparameter tuning.")

        # Define parameter grid for RandomizedSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300, 400],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9, 11],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0, 0.01, 0.1, 1],      # L1 regularization
            'reg_lambda': [0.1, 1, 10]           # L2 regularization
        }

        # Create an XGBRegressor instance
        xgb = XGBRegressor(random_state=42)

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=param_grid,
            n_iter=30,             # Number of different combinations to try
            scoring='neg_root_mean_squared_error',
            cv=5,
            verbose=0,
            n_jobs=-1
        )

        logging.info("Training XGBoost model with hyperparameter tuning.")

        # Fit the model with RandomizedSearchCV
        random_search.fit(X_train, y_train)

        # Best model
        best_xgb = random_search.best_estimator_
        logging.info(f"Best Parameters: {random_search.best_params_}")

        return best_xgb


# Example usage
if __name__ == "__main__":
    # Example DataFrame (replace with actual data loading)
    # df = pd.read_csv('your_data.csv')
    # X_train = df.drop(columns=['target_column'])
    # y_train = df[['target_column']]  # y_train as a DataFrame

    # Example usage of XGBoost Strategy
    # model_builder = ModelBuilder(XGBoostStrategy())
    # trained_model = model_builder.build_model(X_train, y_train)
    # print(trained_model)
    
    pass
