import json

import numpy as np
import pandas as pd
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService


@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    input_data: str,
) -> np.ndarray:

    # Start the service (should be a NOP if already started)
    service.start(timeout=10)
    print('HIIIIII')

    # Load the input data from JSON string
    data = json.loads(input_data)

    # Extract the actual data and expected columns
    data.pop("columns", None)  # Remove 'columns' if it's present
    data.pop("index", None)  # Remove 'index' if it's present

    # Define the columns the model expects
    expected_columns = ["OverallQual",
        "YearBuilt",
        "YearRemodAdd",
        "1stFlrSF",
        "GrLivArea",
        "FullBath",
        "Fireplaces",
        "GarageArea",
        "ExterQual_encoded",
        "Foundation_encoded",
        "HeatingQC_encoded",
        "Neighborhood_encoded",
        "CentralAir_encoded",
        "SaleCondition_encoded",
        "PavedDrive_encoded",
        "LotShape_encoded",
        "SaleType_encoded",
        "HouseStyle_encoded"
    ]


    # Convert the data into a DataFrame with the correct columns
    df = pd.DataFrame(data["data"], columns=expected_columns)

    # Convert DataFrame to JSON list for prediction
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data_array = np.array(json_list)

    # Run the prediction
    prediction = service.predict(data_array)

    return prediction
