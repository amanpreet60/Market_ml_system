import pandas as pd
import os
import zipfile
from abc import ABC, abstractmethod

#creating abstract class for data ingestion

class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        pass

# Implement a concrete class for ZIP Ingestion
class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Extracts a .zip file and returns the content as a pandas DataFrame."""
        # Ensure the file is a .zip
        if not file_path.endswith(".zip"):
            raise ValueError("The provided file is not a .zip file.")

        # Extract the zip file
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall("extracted_data")

        # Find the extracted CSV file (assuming there is one CSV file inside the zip)
        extracted_files = os.listdir("extracted_data")
        csv_files = [f for f in extracted_files if f.endswith(".csv")]
        print(csv_files)
        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV file found in the extracted data.")
        if len(csv_files) > 1:
            raise ValueError("Multiple CSV files found. Please specify which one to use.")

        # Read the CSV into a DataFrame
        csv_file_path = os.path.join("extracted_data", csv_files[0])
        df = pd.read_csv(csv_file_path)

        # Return the DataFrame
        return df

class CsvDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Extracts a .zip file and returns the content as a pandas DataFrame."""
        # Ensure the file is a .zip
        if not file_path.endswith(".csv"):
            raise ValueError("The provided file is not a .csv file.")
  
        # Read the CSV into a DataFrame
        df = pd.read_csv(file_path)

        # Return the DataFrame
        return df

if __name__ == "__main__":
    # # Specify the file path
    file_path = '/Users/amanpreetsingh/My Computer/VSCode/Market/train.zip'
    # # Ingest the data and load it into a DataFrame
    df = ZipDataIngestor().ingest(file_path)

    print(df.head())  # Display the first few rows of the DataFrame
    pass

