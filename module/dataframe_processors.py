import numpy as np
import pandas as pd

def convert_ndarray(df:pd.DataFrame,column:str):
    return np.asarray(df[column],dtype=np.float64)
