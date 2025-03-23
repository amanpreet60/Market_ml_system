from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class univariate_analysis(ABC):
    @abstractmethod
    def my_plot(self, df: pd.DataFrame, feature1: str, feature2: str):
        pass

# for numerical data
class count_plot(univariate_analysis):
    def my_plot(self, df: pd.DataFrame, feature: str):
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()

# for categorical data
class hist_plot(univariate_analysis):
    def my_plot(self, df: pd.DataFrame, feature: str):
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, data=df, palette="muted")
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

if __name__ == "__main__":
    # Example usage of the DataInspector with different strategies.

    # Load the data
    df = pd.read_csv('/Users/amanpreetsingh/Desktop/VSCode/Market/extracted_data/NY-House-Dataset.csv')

    obj1 = count_plot()
    obj1.my_plot(df,'PRICE')

    obj2 = hist_plot()
    obj2.my_plot(df,'TYPE')

    pass
