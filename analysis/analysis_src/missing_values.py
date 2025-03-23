from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd

class missing_value(ABC):
    @abstractmethod
    def get_missing_value(self, df):
        pass
    
