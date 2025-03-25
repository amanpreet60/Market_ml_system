# adding this so that i can import from my other folders : Temporary solution
import sys
sys.path.append('/Users/amanpreetsingh/Desktop/VSCode/Market')

# code starts here
from zenml import step
from src.ingest_data import ZipDataIngestor,CsvDataIngestor
import pandas as pd
 
@step
def data_ingestion_step(file_path: str):
    df = CsvDataIngestor().ingest(file_path)
    return df

