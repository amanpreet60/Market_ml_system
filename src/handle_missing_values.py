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

    def __init__(self,axis=1,thresh=10):
        self.axis = axis
        self.thresh = thresh
        pass

    def handel(self, df):
        logging.info(f"Dropping missing values with axis={self.axis} and thresh={self.thresh}")
        df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)
        logging.info("Missing values dropped.")
        return df_cleaned
    
if __name__ == "__main__":

    df = pd.read_csv('/Users/amanpreetsingh/My Computer/VSCode/Market/extracted_data/updated_housing_data.csv')
    obj1 = remove_missing_value().handel(df)
    print(obj1)