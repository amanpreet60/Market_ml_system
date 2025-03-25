from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class bivariate_analysis(ABC):
    @abstractmethod
    def my_plot(self, df: pd.DataFrame, feature1: str, feature2: str):
        pass

# for numerical data
class scatter_plot(bivariate_analysis):
    def my_plot(self, df: pd.DataFrame, feature1: str, feature2: str):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()

# for categorical/numerical data
class box_plot(bivariate_analysis):
    def my_plot(self, df: pd.DataFrame, feature1: str, feature2: str):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45)
        plt.show()

if __name__ == "__main__":
    # Example usage of the DataInspector with different strategies.

    # Load the data
    df = pd.read_csv('/Users/amanpreetsingh/My Computer/VSCode/Market/extracted_data/NY-House-Dataset.csv')

    obj1 = scatter_plot()
    obj1.my_plot(df,'PRICE','BATH')

    obj2 = box_plot()
    obj2.my_plot(df,'PRICE','BATH')

    pass
