import sys
sys.path.append('/Users/amanpreetsingh/My Computer/VSCode/Market')

# code starts here
from zenml import step
import logging
from src.handle_missing_values import remove_missing_value
import pandas as pd
 
@step(enable_cache=False)
def handle_missing_value_step(df, _method: str = 'remove'):
    if _method == 'remove':
        return remove_missing_value().handel(df)
    else:
        raise ValueError(f"Unsupported missing value handling strategy: {_method}")
    

'''df = pd.read_csv('/Users/amanpreetsingh/My Computer/VSCode/Market/extracted_data/updated_housing_data.csv')
handle_missing_value_step(df)'''