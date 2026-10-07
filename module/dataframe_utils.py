from collections.abc import Sequence, Hashable
import numpy as np
import pandas as pd
from typing import Any, cast
from module.strenums import KeyNames

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

def build_dfs_nu(
    df_long:pd.DataFrame,
    key_nu: str = KeyNames.NU
):
    dfs_nu:dict[float,pd.DataFrame] = {}
    for nu, df in df_long.groupby(key_nu,sort=False):
        df = df.reset_index(drop=True)
        nu = cast(float,nu)
        dfs_nu[nu] = df

    return dfs_nu

def extract_Series_from_idx(
    df: pd.DataFrame,
    idx: int
)->pd.Series:
    return df.loc[idx]

def groupby_to_dict(
    df: pd.DataFrame,
    by: str|Sequence[str],
    *,
    sort: bool = True,
    dropna: bool = True,
) -> dict[Hashable, pd.DataFrame]:
    group_columns = by if isinstance(by, str) else list(by)
    groups: dict[Hashable, pd.DataFrame] = {}

    for key, group in df.groupby(by=by, sort=sort, dropna=dropna):
        grouped = group.copy()
        grouped.attrs = df.attrs.copy()
        groups[key] = grouped

    return groups

def select_maximum_row(
    df: pd.DataFrame,
    column: str,
)-> dict[str,Any]:
    idx = df[column].idxmax()
    return df.loc[[idx]].to_dict(orient="records")[0]
