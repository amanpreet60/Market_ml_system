from abc import ABC, abstractmethod
import pandas as pd

class DataInspection(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        pass

class data_info(DataInspection):
    def inspect(self, df: pd.DataFrame):
        print("\nData Types and Non-null Counts:")
        print(df.info())

class data_describe(DataInspection):
    def inspect(self, df: pd.DataFrame):
        print("\nSummary Statistics (Numerical Features):")
        print(df.describe())
        print("\nSummary Statistics (Categorical Features):")
        print(df.describe(include=["O"]))

if __name__ == "__main__":
    # Example usage of the DataInspector with different strategies.

    # Load the data
    df = pd.read_csv('/Users/amanpreetsingh/My Computer/VSCode/Market/extracted_data/NY-House-Dataset.csv')

    obj1 = data_info()
    obj1.inspect(df)

    obj2 = data_describe()
    obj2.inspect(df)

    pass
