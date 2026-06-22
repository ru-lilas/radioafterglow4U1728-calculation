from numpy.typing import NDArray
import pandas as pd
import numpy as np

def revive_quantity_array(
    dimensionless_value: NDArray[np.float64],
    quantity_as_unit: NDArray[np.float64]
)->NDArray[np.float64]:
    return dimensionless_value*quantity_as_unit

def calcualte_product_two_columns(
    df: pd.DataFrame,
    columns: tuple[str,str],
)->NDArray[np.float64]:
    arrs = [
        np.asarray(df[column],dtype=np.float64)
        for column in columns
    ]
    return arrs[0]*arrs[1]

def fetch_reviving_quantity(
    df: pd.DataFrame,
    column_dimless: str,
    column_norm: str
)->tuple[NDArray[np.float64],NDArray[np.float64]]:
    dimless = np.asarray(df[column_dimless],dtype=np.float64)
    norm = np.asarray(df[column_norm],dtype=np.float64)
    return dimless,norm

def build_axis_array(conf:dict,df:pd.DataFrame):
    if "column" in conf.keys():
        column_name = str(conf["column"])
        return np.asarray(df[column_name],dtype=np.float64)
    else:
        column_dimless = str(conf["column_dimensionless"])
        column_norm = str(conf["column_normalization"])
        dimless_arr, norm_arr = fetch_reviving_quantity(df,column_dimless,column_norm)
        return revive_quantity_array(dimless_arr,norm_arr)
