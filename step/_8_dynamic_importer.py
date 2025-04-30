import pandas as pd
from zenml import step


@step
def dynamic_importer() -> str:
    """Dynamically imports data for testing out the model."""
    # Here, we simulate importing or generating some data.
    # In a real-world scenario, this could be an API call, database query, or loading from a file.
    data = {
    "Order": [1, 2],
    "PID": [1461, 1462],
    "OverallQual": [5, 6],
    "YearBuilt": [1961, 1958],
    "YearRemodAdd": [1961, 1958],
    "1stFlrSF": [896, 1329],
    "GrLivArea": [896, 1329],
    "FullBath": [1, 1],
    "Fireplaces": [0, 0],
    "GarageArea": [730, 312],
    "ExterQual": ["TA", "TA"],
    "Foundation": ["CBlock", "CBlock"],
    "HeatingQC": ["TA", "TA"],
    "Neighborhood": ["NAmes", "NAmes"],
    "CentralAir": ["Y", "Y"],
    "SaleCondition": ["Normal", "Normal"],
    "PavedDrive": ["Y", "Y"],
    "LotShape": ["Reg", "IR1"],
    "SaleType": ["WD", "WD"],
    "HouseStyle": ["1Story", "1Story"],
}

    df = pd.DataFrame(data)

    # Convert the DataFrame to a JSON string
    json_data = df.to_json(orient="split")

    return json_data
