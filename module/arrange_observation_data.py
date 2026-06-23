from typing import Any,cast
import pandas as pd
import numpy as np
import astropy.units as u
from module import quantity_converter

def extract_df_before_xrb(df:pd.DataFrame)->pd.DataFrame:
    mask:list[bool] = df["t"] < 0.0
    return pd.DataFrame(df[mask])

def calculate_time_avaraged_flux(df:pd.DataFrame,column:str):
    arr = np.asarray(df[column],dtype=np.float64)
    return float(arr.mean())

def calculate_net_flux(
    df:pd.DataFrame,
    column_flux:str,
    flux_per: float,
):
    flux_arr = np.asarray(df[column_flux],dtype=np.float64)
    return np.maximum(flux_arr - flux_per, 0.0)

def extract_peak_flux(df:pd.DataFrame,column_flux:str):
    idx_peak = cast(np.int64, df[column_flux].idxmax())
    t_peak = float(df.at[idx_peak, "t"])
    f_peak = float(df.at[idx_peak, column_flux])
    return (t_peak,f_peak)

def arrange_df_for_band(
    df:pd.DataFrame,
    df_before_xrb:pd.DataFrame,
    metadata:dict[str,Any],
    flux_column:str
):
    flux_per = calculate_time_avaraged_flux(df_before_xrb,flux_column)
    flux_net = calculate_net_flux(df,flux_column,flux_per)
    df[f"{flux_column}_net"] = flux_net

    t_peak, flux_peak = extract_peak_flux(df,f"{flux_column}_net")
    metadata[f"{flux_column}_per"] = flux_per
    metadata[f"{flux_column}_peak_time"] = t_peak
    metadata[f"{flux_column}_peak_net"] = flux_peak
    return

def build_arranged_df(metadata:dict[str,Any],df:pd.DataFrame):
    df_before_xrb = extract_df_before_xrb(df)
    arrange_df_for_band(df,df_before_xrb,metadata,"f5")
    arrange_df_for_band(df,df_before_xrb,metadata,"f9")

def calculate_phi_peak(
    nu_value:float,
    nu_unit:str,
    t_peak_value:float,
    t_unit:str,
    phi_unit:str
)->float:
    nu_quantity = u.Quantity(nu_value,nu_unit)
    t_quantity = u.Quantity(t_peak_value,t_unit)
    phi_quantity = u.Quantity((nu_quantity*t_quantity))
    phi_quantity = phi_quantity.to(u.Unit(phi_unit))
    return float(phi_quantity.value)

def convert_fnu_into_lnu(
    fnu_value:float,
    fnu_unit:str,
    d_value:float,
    d_unit: str,
    l_unit: str
):
    fnu_quantity = u.Quantity(fnu_value,fnu_unit)
    d_quantity = u.Quantity(d_value,d_unit)
    l_quantity = quantity_converter.flux_into_luminosity(
        flux=fnu_quantity,
        distance=d_quantity
    )
    l_quantity = l_quantity.to(u.Unit(l_unit))
    return float(l_quantity.value)
