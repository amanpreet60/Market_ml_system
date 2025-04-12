import sys
sys.path.append('/Users/amanpreetsingh/My Computer/VSCode/Market')

import pandas as pd

from step._1_data_ingestion_step import data_ingestion_step
from step._2_handle_missing_value_step import handle_missing_value_step
from step._3_outlier_detection_step import outlier_detection_step
from step._4_data_splitter_step import data_splitter_step
from step._5_feature_engineering import feature_engineering_step

'''from step.data_splitter_step import data_splitter_step
from step.feature_engineering_step import feature_engineering_step
from step.model_building_step import model_building_step
from step.model_evaluator_step import model_evaluator_step
'''

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
        file_path="/Users/amanpreetsingh/My Computer/VSCode/Market/extracted_data/NY-House-Dataset.csv"
    )

    # Handling Missing Values Step
    filled_data = handle_missing_value_step(raw_data)

    # Outlier Detection Step
    clean_data = outlier_detection_step(filled_data, column_name="PRICE",eng_type = 'iqr')

    #Data Splitting Step
    X_train, X_test, y_train, y_test = data_splitter_step(clean_data, target_column="PRICE")

    # Feature Engineering Step
    engineered_data = feature_engineering_step(
        y_train, strategy="log", features=["PRICE"]
    )

    '''


    # Model Building Step
    model = model_building_step(X_train=X_train, y_train=y_train)

    # Model Evaluation Step
    evaluation_metrics, mse = model_evaluator_step(
        trained_model=model, X_test=X_test, y_test=y_test
    )'''

    return


if __name__ == "__main__":
    # Running the pipeline
    run = ml_pipeline()
