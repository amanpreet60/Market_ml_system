#check outlier 

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import skew, kurtosis

class handel_outliers(ABC):
    @abstractmethod
    def check_skew_kurt(self, df: pd.DataFrame, feature: str):
        pass

class feature_skew_kurt(handel_outliers):
    def check_skew_kurt(self, df, feature: str):
        skewness = skew(df[feature])
        kurt = kurtosis(df[feature])

        return [float(skewness),float(kurt)]
    
    def process_feature(self, skewness, kurtosis):

        # Skewness Interpretation
        if abs(skewness) < 0.5:
            skewness_interpret = "Symmetric (Normal Distribution)"
        elif skewness > 0:
            skewness_interpret = "Right-Skewed"
        else:
            skewness_interpret = "Left-Skewed"

        # Kurtosis Interpretation
        if abs(kurtosis - 3) < 0.5:
            kurtosis_interpret = "Normal Distribution"
        elif kurtosis > 3:
            kurtosis_interpret = "Heavy-tailed (Laplace/T-Distribution)"
        else:
            kurtosis_interpret = "Light-tailed (Uniform Distribution)"

        print(skewness_interpret)
        print(kurtosis_interpret)
        print(".\n.\n.\n.")
        print("Z-Score: Best for normally distributed data.\nIQR: Works well for skewed..."
        " distributions and robust datasets.\nPercentile Capping: Useful when you don't want to lose data.")
    
    def z_score(self, df: pd.DataFrame, feature: str):

        df['z_score'] = (df[feature] - df[feature].mean()) / df[feature].std()

        # Keep only rows where Z-Score is within ±3 standard deviations
        df_filtered = df[abs(df['z_score']) < 3]
        print('Outlier exist\nDetected using z-score with threshold 3')

        return df_filtered
    
    def iqr(self, df: pd.DataFrame, feature: str):

        # Compute IQR
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1

        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter data
        df_filtered = df[(df[feature] < lower_bound) & (df[feature] > upper_bound)]
        print("Outliers detected using the IQR method.")

        return df_filtered
    
    def winsorization(self, df: pd.DataFrame, feature: str):

        # Set percentile limits
        lower_limit = df[feature].quantile(0.05)  # 5th percentile
        upper_limit = df[feature].quantile(0.95)  # 95th percentile

        # Cap values
        df_filtered = df[(df[feature] < lower_limit) & (df[feature] > upper_limit)]
        print("Outliers detected using the winsorization method.")
        return df_filtered

        
if __name__ == '__main__':

    df = pd.read_csv('/Users/amanpreetsingh/Desktop/VSCode/Market/extracted_data/NY-House-Dataset.csv')

    obj1 = feature_skew_kurt()
    temp = obj1.check_skew_kurt(df,'PRICE')
    print(temp)
    obj1.process_feature(temp[0],temp[1])

    df = obj1.winsorization(df,'PRICE')