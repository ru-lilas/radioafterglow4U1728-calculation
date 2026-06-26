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

def extract_df_largest_sample(
    df:pd.DataFrame,
    column_y:str,
    n_sample: int
):
    return df.nlargest(n_sample,column_y)

def extract_largest2_mean(
    df:pd.DataFrame,
    column_x:str,
    column_y:str
):
    df_peak = extract_df_largest_sample(df,column_y,2)
    x_peak = float(convert_ndarray(df_peak,column_x).mean())
    y_peak = float(convert_ndarray(df_peak,column_y).mean())
    return (x_peak,y_peak)

def extract_peak_quadratic(
    df: pd.DataFrame,
    column_x:str,
    column_y:str,
    column_yerr: str,
    n_sample: int
):
    df_sample = extract_df_largest_sample(df,column_y,n_sample)
    x = convert_ndarray(df_sample,column_x)
    y = convert_ndarray(df_sample,column_y)
    yerr = convert_ndarray(df_sample,column_yerr)
    coef = np.polyfit(x, y, deg=2, w=1.0 / yerr)
    a, b, c = coef

    x_peak = -b / (2 * a)
    y_peak = a * x_peak**2 + b * x_peak + c
    return (x_peak,y_peak)
