import sys
sys.path.append('/Users/amanpreetsingh/My Computer/VSCode/Market')

import logging

import pandas as pd
from src.handle_outliers import feature_skew_kurt
from zenml import step


@step(enable_cache=False)
def outlier_detection_step(df: pd.DataFrame, column_name: str, eng_type: str = 'iqr') -> pd.DataFrame:
    """Detects and removes outliers using OutlierDetector."""
    logging.info(f"Starting outlier detection step with DataFrame of shape: {df.shape}")

    if df is None:
        logging.error("Received a NoneType DataFrame.")
        raise ValueError("Input df must be a non-null pandas DataFrame.")

    if not isinstance(df, pd.DataFrame):
        logging.error(f"Expected pandas DataFrame, got {type(df)} instead.")
        raise ValueError("Input df must be a pandas DataFrame.")

    if column_name not in df.columns:
        logging.error(f"Column '{column_name}' does not exist in the DataFrame.")
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
        # Ensure only numeric columns are passed
    df_numeric = df.select_dtypes(include=[int, float])

    analyse = feature_skew_kurt()
    print(analyse.check_skew_kurt(df,column_name))
    if eng_type=='z_test':
        return analyse.z_score(df,column_name)
    elif eng_type == 'iqr':
        return analyse.iqr(df,column_name)
    elif eng_type == 'cap':
        return analyse.winsorization(df,column_name)
    else:
        logging.error("Wrong Input")


'''df = pd.read_csv('/Users/amanpreetsingh/My Computer/VSCode/Market/extracted_data/updated_housing_data.csv')
outlier_detection_step(df,'SalePrice')
'''