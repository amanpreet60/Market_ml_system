import sys
sys.path.append('/Users/amanpreetsingh/My Computer/VSCode/Market')

import pandas as pd

from step._1_data_ingestion_step import data_ingestion_step
from step._2_handle_missing_value_step import handle_missing_value_step
from step._3_outlier_detection_step import outlier_detection_step
from step._4_data_splitter_step import data_splitter_step
from step._5_feature_engineering import feature_engineering_step
from step._6_model_building_step import model_building_step
from step._7_model_evaluator_step import model_evaluator_step
import logging

from zenml import Model, pipeline, step, model


@pipeline(
    model=Model(
        # The name uniquely identifies this model
        name="prices_predictor"
    ),
)
def ml_pipeline():
    """Define an end-to-end machine learning pipeline."""

    # Data Ingestion Step
    raw_data = data_ingestion_step(
        file_path="/Users/amanpreetsingh/My Computer/VSCode/Market/extracted_data/updated_housing_data.csv"
    )
    # Handling Missing Values Step
    filled_data = handle_missing_value_step(raw_data)
    print(filled_data)
    # Outlier Detection Step
    clean_data = outlier_detection_step(filled_data, column_name="SalePrice",eng_type = 'iqr')

    #Data Splitting Step
    X_train, X_test, y_train, y_test = data_splitter_step(clean_data, target_column="SalePrice")

    X_train, X_test, y_train, y_test = feature_engineering_step(X_train, X_test, y_train, y_test, 'log',['GrLivArea'])
    X_train, X_test, y_train, y_test = feature_engineering_step(X_train, X_test, y_train, y_test, 'label_encoding',['ExterQual', 'Foundation', 
                                                                                                                    'HeatingQC', 'Neighborhood',
                                                                                                                      'CentralAir', 'SaleCondition', 
                                                                                                                      'PavedDrive', 'LotShape', 
                                                                                                                      'SaleType', 'HouseStyle'])

    logging.info(X_train)
    # Model Building Step
    model = model_building_step(X_train=X_train, y_train=y_train)

    # Model Evaluation Step
    evaluation_metrics, mse = model_evaluator_step(
        trained_model=model, X_test=X_test, y_test=y_test
    )
    
    return model


if __name__ == "__main__":
    # Running the pipeline
    run = ml_pipeline()
