import numpy as np
import pandas as pd
from typing import cast

def convert_ndarray(df:pd.DataFrame,column:str):
    return np.asarray(df[column],dtype=np.float64)

def extract_maximum(df:pd.DataFrame,column_x:str,column_y:str):
    idx_peak = cast(np.int64, df[column_y].idxmax())
    x_peak = float(df.at[idx_peak, column_x])
    y_peak = float(df.at[idx_peak, column_y])
    return (x_peak,y_peak)

def extract_largest2_mean(
    df:pd.DataFrame,
    column_x:str,
    column_y:str
):
    df_peak  = df.nlargest(2,column_y)
    x_peak = float(convert_ndarray(df_peak,column_x).mean())
    y_peak = float(convert_ndarray(df_peak,column_y).mean())
    return (x_peak,y_peak)
