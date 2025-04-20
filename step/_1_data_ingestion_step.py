import sys
sys.path.append('/Users/amanpreetsingh/My Computer/VSCode/Market')

# code starts here
from zenml import step
from src.ingest_data import ZipDataIngestor,CsvDataIngestor
import pandas as pd
 
@step(enable_cache=False)
def data_ingestion_step(file_path: str):
    df = CsvDataIngestor().ingest(file_path)
    return df

