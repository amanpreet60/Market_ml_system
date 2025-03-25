from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd

class missing_value(ABC):
    @abstractmethod
    def get_missing_value(self, df):
        pass

class identify_missing_value(missing_value):
    def get_missing_value(self, df):
        print("\nMissing Values Count by Column:")
        missing_values = df.isnull().sum()
        print(missing_values)
        pass
    
if __name__ == "__main__":

    # Load the data
    df = pd.read_csv('/Users/amanpreetsingh/My Computer/VSCode/Market/extracted_data/NY-House-Dataset.csv')

    # Perform Missing Values Analysis
    obj1 = identify_missing_value()
    obj1.get_missing_value(df)