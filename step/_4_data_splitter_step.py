import sys
sys.path.append('/Users/amanpreetsingh/My Computer/VSCode/Market')
from typing import Tuple

import pandas as pd
from src.data_split import SimpleTrainTestSplit
from zenml import step


@step(enable_cache=False)
def data_splitter_step(
    df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits the data into training and testing sets using DataSplitter."""
    splitter = SimpleTrainTestSplit()
    X_train, X_test, y_train, y_test = splitter.split_data(df, target_column)
    return X_train, X_test, pd.DataFrame(y_train, columns=[target_column]), pd.DataFrame(y_test, columns=[target_column])
