import logging
from abc import ABC, abstractmethod

import pandas as pd

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class handel_missing_values_abc(ABC):
    @abstractmethod
    def handel(self, df: pd.DataFrame):
        pass

class remove_missing_value(handel_missing_values_abc):

    def __init__(self,axis=0,thresh=None):
        self.axis = axis
        self.thresh = thresh
        pass

    def handel(self, df):
        logging.info(f"Dropping missing values with axis={self.axis} and thresh={self.thresh}")
        df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)
        logging.info("Missing values dropped.")
        return df_cleaned

class fill_missing_values(handel_missing_values_abc):

    def __init__(self, method="mean", fill_value=None):
        self.method = method
        self.fill_value = fill_value
        pass

    def handel(self,df):

        logging.info(f"Filling missing values using method: {self.method}")

        df_cleaned = df.copy()
        if self.method == "mean":
            numeric_columns = df_cleaned.select_dtypes(include="int64").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].mean()
            )
        elif self.method == "median":
            numeric_columns = df_cleaned.select_dtypes(include="int64").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].median()
            )
        elif self.method == "mode":
            for column in df_cleaned.columns:
                df_cleaned[column].fillna(df[column].mode().iloc[0], inplace=True)
        elif self.method == "constant":
            df_cleaned = df_cleaned.fillna(self.fill_value)
        else:
            logging.warning(f"Unknown method '{self.method}'. No missing values handled.")

        logging.info("Missing values filled.")
        return df_cleaned
    
if __name__ == "__main__":

    df = pd.read_csv('/Users/amanpreetsingh/My Computer/VSCode/Market/extracted_data/NY-House-Dataset.csv')
    obj1 = remove_missing_value().handel(df)
    

    obj2 = fill_missing_values(method='mean', fill_value=None)
    obj2.handel(df)