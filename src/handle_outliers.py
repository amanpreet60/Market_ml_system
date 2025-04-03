#check outlier 

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
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

        logging.info('Skewness: ',skewness_interpret)
        logging.info('Kurtosis: ',kurtosis_interpret)
    
    def z_score(self, df: pd.DataFrame, feature: str):

        df['z_score'] = (df[feature] - df[feature].mean()) / df[feature].std()

        # Keep only rows where Z-Score is within ±3 standard deviations
        df_filtered = df[abs(df['z_score']) > 3]
        print('Outlier exist\nDetected using z-score with threshold 3')

        return df_filtered
    
    def iqr(self, df: pd.DataFrame, feature: str):

        # Compute IQR
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1

        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        print(lower_bound)
        upper_bound = Q3 + 1.5 * IQR
        print(upper_bound)

        # Filter data
        df_filtered = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
        print(df_filtered)
        print("Outliers detected using the IQR method.")

        return df_filtered
    
    def winsorization(self, df: pd.DataFrame, feature: str):

        # Set percentile limits
        lower_limit = df[feature].quantile(0.05)  # 5th percentile
        upper_limit = df[feature].quantile(0.95)  # 95th percentile

        # Cap values
        df_filtered = df[(df[feature] >= lower_limit) & (df[feature] <= upper_limit)]
        print("Outliers detected using the winsorization method.")
        return df_filtered
    
    def visualize_outliers(self, df: pd.DataFrame, features: list):
        logging.info(f"Visualizing outliers for features: {features}")
        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[feature])
            plt.title(f"Boxplot of {feature}")
            plt.show()
        logging.info("Outlier visualization completed.")

        
if __name__ == '__main__':

    df = pd.read_csv('/Users/amanpreetsingh/My Computer/VSCode/Market/extracted_data/NY-House-Dataset.csv')
    print(df.shape)
    obj1 = feature_skew_kurt()
    temp = obj1.check_skew_kurt(df,'BEDS')
    print(temp)
    obj1.process_feature(temp[0],temp[1])
    df = obj1.iqr(df,'BEDS')
    print(df.shape)

    obj1.visualize_outliers(df,['PRICE','BEDS','BATH'])
    temp = obj1.check_skew_kurt(df,'BEDS')
    obj1.process_feature(temp[0],temp[1])

