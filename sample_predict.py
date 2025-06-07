import requests

url = "http://127.0.0.1:8002/invocations"

input_data = {
    "dataframe_records": [
        {
            "OverallQual": 7,
            "YearBuilt": 2006,
            "YearRemodAdd": 2007,
            "1stFlrSF": 625,
            "GrLivArea": 7.131699,
            "FullBath": 2,
            "Fireplaces": 0,
            "GarageArea": 625,
            "ExterQual_encoded": 2,
            "Foundation_encoded": 2,
            "HeatingQC_encoded": 0,
            "Neighborhood_encoded": 21,
            "CentralAir_encoded": 1,
            "SaleCondition_encoded": 4,
            "PavedDrive_encoded": 2,
            "LotShape_encoded": 0,
            "SaleType_encoded": 8,
            "HouseStyle_encoded": 5,
        }
    ]
}

response = requests.post(url, json=input_data)

if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print(f"Error: {response.status_code}")
    print(response.text)
