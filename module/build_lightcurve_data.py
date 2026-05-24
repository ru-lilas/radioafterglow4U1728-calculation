# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false
import astropy.units as u
import numpy as np
import pandas as pd
from pathlib import Path
from module.utilities import filereaders as fr

def create_reference_frequency_list(config:dict):
    unit = u.Unit(config["unit"])
    values = np.asarray(config["values"])
    quantity = u.Quantity(values,unit)
    return np.asarray(quantity.to_value(u.Hz))

def _fetch_lc_row(df:pd.DataFrame,band_center:float):
    row = df.iloc[(df["nu"] - band_center).abs().argmin()]
    return row

def _build_lc_single_line(metadata:dict,df:pd.DataFrame,nu:float):
    t = float(metadata["t_value"])
    row = _fetch_lc_row(df,nu)
    return float(row["nu"]),{
        "t":t,
        "lnu": float(row["lnu"]),
    }

def build_lightcurve_data(inpath_list:list[Path],nu_ref:float)->pd.DataFrame:
    lc_longformat = []
    for inpath in inpath_list:
        metadata = fr.read_keyvalue(inpath)
        df = fr.read_csv(inpath)
        nu,lc_row = _build_lc_single_line(metadata,df,nu_ref)
        lc_single_line = {
            "nu": nu,
            **metadata,
            **lc_row
        }
        lc_longformat.append(lc_single_line)

    return pd.DataFrame(lc_longformat)
