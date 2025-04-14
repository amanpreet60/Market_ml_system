# adding this so that i can import from my other folders : Temporary solution
import sys
sys.path.append('/Users/amanpreetsingh/My Computer/VSCode/Market')
import pandas as pd
from src.feature_engineering import (
    LogTransformation,
    MinMaxScaling,
    OneHotEncoding,
    StandardScaling,
    drop,
)
from zenml import step


@step
def feature_engineering_step(
    df: pd.DataFrame, strategy: str = "log", features: list = None
) -> pd.DataFrame:
    """Performs feature engineering using FeatureEngineer and selected strategy."""

    # Ensure features is a list, even if not provided
    if features is None:
        features = []  # or raise an error if features are required

    if strategy == "log":
        engineer = LogTransformation(features)
    elif strategy == "standard_scaling":
        engineer = StandardScaling(features)
    elif strategy == "minmax_scaling":
        engineer = MinMaxScaling(features)
    elif strategy == "onehot_encoding":
        engineer = OneHotEncoding(features)
    elif strategy == 'drop':
        engineer = drop(features)
    else:
        raise ValueError(f"Unsupported feature engineering strategy: {strategy}")

    transformed_df = engineer.apply_transformation(df)
    return transformed_df
