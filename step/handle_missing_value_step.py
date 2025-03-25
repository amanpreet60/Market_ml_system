# adding this so that I can import from my other folders : Temporary solution
import sys
sys.path.append('/Users/amanpreetsingh/Desktop/VSCode/Market')

# code starts here
from zenml import step
import logging
from src.handle_missing_values import fill_missing_values,remove_missing_value
import pandas as pd
 
@step
def handle_missing_value_step(df, _method: str = 'mean'):
    if _method == 'fill':
        return remove_missing_value().handel(df)
    elif _method in ['mean','median','mode','constant']:
        return fill_missing_values(method=_method,fill_value=None).handel(df)
    else:
        raise ValueError(f"Unsupported missing value handling strategy: {_method}")
    

'''df = pd.read_csv('/Users/amanpreetsingh/Desktop/VSCode/Market/extracted_data/NY-House-Dataset.csv')
hande_missing_value_step(df)'''