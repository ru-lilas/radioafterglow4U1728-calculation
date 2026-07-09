import numpy as np
import pandas as pd
from typing import cast

def extract_column_as_ndarray(df:pd.DataFrame,column:str):
    return np.asarray(df[column],dtype=np.float64)

def build_dfs_grouped(
    df_long:pd.DataFrame,
    group_by: str
):
    dfs:list[tuple[float,pd.DataFrame]] = []
    for group_key, df in df_long.groupby(group_by,sort=False):
        df = df.reset_index(drop=True)
        group_key = cast(float,group_key)
        dfs.append((group_key,df))

    return dfs

def extract_minimum_row(
    df: pd.DataFrame,
    column: str
):
    row = cast(pd.Series,df.loc[df[column].idxmin()])
    return row
