import logging
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.model_selection import train_test_split

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataSplitting(ABC):
    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str):
        pass

class SimpleTrainTestSplit(DataSplitting):
    def __init__(self, test_size=0.2, random_state=42):

        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, target_column: str):

        logging.info("Performing simple train-test split.")
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        logging.info("Train-test split completed.")
        return X_train, X_test, y_train, y_test


# Example usage
if __name__ == "__main__":
    # Example dataframe (replace with actual data loading)
    df = pd.read_csv('/Users/amanpreetsingh/My Computer/VSCode/Market/extracted_data/NY-House-Dataset.csv')

    # Initialize data splitter with a specific strategy
    data_splitter = SimpleTrainTestSplit(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = data_splitter.split_data(df, target_column='PRICE')
    pass
