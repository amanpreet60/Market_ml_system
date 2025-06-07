import pandas as pd
from zenml import step


@step
def dynamic_importer() -> str:
    """Dynamically imports data for testing out the model."""
    # Here, we simulate importing or generating some data.
    # In a real-world scenario, this could be an API call, database query, or loading from a file.
    data = {
    "OverallQual": [7, 5],
    "YearBuilt": [2006, 1966],
    "YearRemodAdd": [2007, 1966],
    "1stFlrSF": [625, 955],
    "GrLivArea": [7.131699, 6.862758],
    "FullBath": [2, 1],
    "Fireplaces": [0, 0],
    "GarageArea": [625, 386],
    "ExterQual": [2, 3],
    "Foundation": [2, 1],
    "HeatingQC": [0, 0],
    "Neighborhood": [21, 19],
    "CentralAir": [1, 1],
    "SaleCondition": [4, 4],
    "PavedDrive": [2, 2],
    "LotShape": [0, 3],
    "SaleType": [8, 8],
    "HouseStyle": [5, 2],
}


    df = pd.DataFrame(data)

    # Convert the DataFrame to a JSON string
    json_data = df.to_json(orient="split")

    return json_data
