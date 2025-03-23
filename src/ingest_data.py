import pandas as pd
import os
import zipfile
from abc import ABC, abstractmethod

#creating abstract class for data ingestion

class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        pass
