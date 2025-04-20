import logging
# adding this so that i can import from my other folders : Temporary solution
import sys
sys.path.append('/Users/amanpreetsingh/My Computer/VSCode/Market')

from sklearn.model_selection import train_test_split
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from abc import ABC, abstractmethod

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class FeatureEngineering(ABC):
    @abstractmethod
    def apply_transformation(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> tuple:
        pass


class LogTransformation(FeatureEngineering):
    def __init__(self, features, apply_to_target=False):
        self.features = features
        self.apply_to_target = apply_to_target  # Whether to transform y as well

    def apply_transformation(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame)-> tuple:
        logging.info(f"Applying log transformation to features: {self.features}")
        X_train_transformed = X_train.copy()
        X_test_transformed = X_test.copy()
        y_train_transformed = y_train.copy()
        y_test_transformed = y_test.copy()

        for feature in self.features:
            X_train_transformed[feature] = np.log1p(X_train[feature])
            X_test_transformed[feature] = np.log1p(X_test[feature])

        if self.apply_to_target:
            logging.info("Also applying log transformation to target.")
            y_train_transformed = np.log1p(y_train)
            y_test_transformed = np.log1p(y_test)
        logging.info("Log transformation completed.")
        return X_train_transformed, X_test_transformed, y_train_transformed, y_test_transformed


class LabelEncoding(FeatureEngineering):
    def __init__(self, features: list):
        self.features = features
        self.encoders = {}

    def apply_transformation(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> tuple:
        logging.info(f"Applying label encoding to features: {self.features}")
        
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()
        
        for feature in self.features:
            le = LabelEncoder()
            le.fit(X_train[feature].astype(str))  # Fit only on train
            self.encoders[feature] = le

            # Encode train
            X_train_encoded[f"{feature}_encoded"] = le.transform(X_train[feature].astype(str))

            # Encode test with handling unseen labels
            known_classes = set(le.classes_)
            X_test_encoded[f"{feature}_encoded"] = X_test[feature].astype(str).apply(
                lambda x: le.transform([x])[0] if x in known_classes else -1
            )
            # Drop original feature after encoding
            X_train_encoded.drop(columns=[feature], inplace=True)
            X_test_encoded.drop(columns=[feature], inplace=True)

        logging.info("Label encoding completed.")
        return X_train_encoded, X_test_encoded, y_train, y_test


class Drop(FeatureEngineering):
    def __init__(self, features):
        self.features = features

    def apply_transformation(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> tuple:
        logging.info(f"Dropping features: {self.features}")
        
        X_train_dropped = X_train.drop(columns=self.features)
        X_test_dropped = X_test.drop(columns=self.features)
        
        logging.info("Dropping completed.")
        return X_train_dropped, X_test_dropped, y_train, y_test


# Example Usage
if __name__ == "__main__":
    # Example dataframe (replace with actual data)
    df = pd.read_csv('/Users/amanpreetsingh/My Computer/VSCode/Market/extracted_data/updated_housing_data.csv')
    
    # Split your data
    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply transformations
    log_transform = LogTransformation(features=["GrLivArea"], apply_to_target=True)
    X_train_log, X_test_log, y_train_log, y_test_log = log_transform.apply_transformation(X_train, X_test, y_train, y_test)

    label_encoder = LabelEncoding(features=["ExterQual"])
    X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = label_encoder.apply_transformation(X_train_log, X_test_log, y_train_log, y_test_log)

    #drop_columns = Drop(features=["LotConfig"])
    #X_train_final, X_test_final, y_train_final, y_test_final = drop_columns.apply_transformation(X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded)
    
    # Check final transformed data
    print(X_train_encoded.head())
    print(y_train_encoded.head())
