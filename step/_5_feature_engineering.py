import sys
sys.path.append('/Users/amanpreetsingh/My Computer/VSCode/Market')
import pandas as pd
from src.feature_engineering import (
    LogTransformation,
    LabelEncoding,
    Drop
)
from zenml import step

@step
def feature_engineering_step(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    strategy: str = "log",
    features: list = None,
) -> tuple:
    """Performs feature engineering using selected strategy on X_train, X_test, y_train, and y_test."""

    # Ensure features is a list, even if not provided
    if features is None:
        features = []  # or raise an error if features are required

    # Apply appropriate transformation strategy
    if strategy == "log":
        engineer = LogTransformation(features, apply_to_target=True)
    elif strategy == "label_encoding":
        engineer = LabelEncoding(features)
    elif strategy == 'drop':
        engineer = Drop(features)
    else:
        raise ValueError(f"Unsupported feature engineering strategy: {strategy}")

    # Apply transformation to X_train, X_test, y_train, y_test
    X_train_transformed, X_test_transformed, y_train_transformed, y_test_transformed = engineer.apply_transformation(
        X_train, X_test, y_train, y_test
    )

    # Return the transformed data
    return X_train_transformed, X_test_transformed, y_train_transformed, y_test_transformed
