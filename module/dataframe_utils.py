import numpy as np
import pandas as pd
from typing import cast

def extract_column_as_ndarray(df:pd.DataFrame,column:str):
    return np.asarray(df[column],dtype=np.float64)
