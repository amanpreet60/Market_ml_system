
import sys
sys.path.append('/Users/amanpreetsingh/My Computer/VSCode/Market')

import logging
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

#Abstract Base Class for Model Evaluation Strategy
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict:
        pass


# Concrete Strategy for Regression Model Evaluation
class RegressionModelEvaluation(ModelEvaluationStrategy):
    def evaluate_model(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.DataFrame
    ) -> dict:
        logging.info("Predicting using the trained model.")
        y_pred = model.predict(X_test)
        y_pred = np.expm1(y_pred)
        y_test = np.expm1(y_test)
        y_pred = pd.DataFrame(y_pred, columns=['SalePrice'])
        logging.info("Calculating evaluation metrics.")
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        metrics = {"Root Mean Squared Error": rmse, "R-Squared": r2}

        logging.info(f"Model Evaluation Metrics: {metrics}")
        return metrics

# Example usage
if __name__ == "__main__":
    # Example trained model and data (replace with actual trained model and data)
    # model = trained_sklearn_model
    # X_test = test_data_features
    # y_test = test_data_target

    # Initialize model evaluator with a specific strategy
    # model_evaluator = ModelEvaluator(RegressionModelEvaluationStrategy())
    # evaluation_metrics = model_evaluator.evaluate(model, X_test, y_test)
    # print(evaluation_metrics)

    pass
