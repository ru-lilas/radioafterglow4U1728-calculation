import pandas as pd
from typing import Any
from numpy.typing import NDArray
from module.models import InputParameters
import numpy as np
import astropy.units as u
from itertools import product
from tqdm import tqdm

def beta(inputs:dict):
    beta_arr:NDArray[np.float64] = \
        np.logspace(**inputs["beta_arr"],dtype=np.float64)
    inputparams:list[InputParameters] = []
    for beta in beta_arr:
        inputparams.append(InputParameters(
            **inputs["fixed"],
            beta_sh=beta
        ))
    return inputparams

def awind(inputs:dict):
    a_wind_arr:NDArray[np.float64] = \
        np.logspace(**inputs["a_wind_arr"],dtype=np.float64)
    inputparams:list[InputParameters] = []
    for a_wind in a_wind_arr:
        inputparams.append(InputParameters(
            **inputs["fixed"],
            a_wind_value=a_wind
        ))
    return inputparams

def both(inputs:dict):
    a_wind_arr:NDArray[np.float64] = \
        np.logspace(**inputs["a_wind_arr"],dtype=np.float64)
    beta_arr:NDArray[np.float64] = \
        np.logspace(**inputs["beta_arr"],dtype=np.float64)
    inputparams:list[InputParameters] = []
    for beta in beta_arr:
        for a_wind in a_wind_arr:
            inputparams.append(InputParameters(
                **inputs["fixed"],
                beta_sh=beta,
                a_wind_value=a_wind
            ))
    return inputparams

def _table_row(
    beta_sh: float,
    a_wind_value: float,
    fixed: dict[str, Any],
    units_designated: dict[str,str]
) -> dict[str,Any]:

    params = InputParameters(
        **fixed,
        beta_sh=beta_sh,
        a_wind_value=a_wind_value,
    )
    phi_unit = u.Unit(units_designated["phi_unit"])
    l_unit = u.Unit(units_designated["l_unit"])
    phi_theta_value = params.phi_theta.to_value(phi_unit)
    l_theta_value = params.l_theta.to_value(l_unit)

    return {
        "beta_sh": beta_sh,
        "a_wind_value": a_wind_value,
        "phi_theta_value": phi_theta_value,
        "l_theta_value": l_theta_value,
        "tau_theta": params.tau_theta,
    }

def table(
    config_data:dict[str,Any]
)->pd.DataFrame:
    a_wind_arr:NDArray[np.float64] = \
        np.logspace(**config_data["a_wind_arr"],dtype=np.float64)
    beta_arr:NDArray[np.float64] = \
        np.logspace(**config_data["beta_arr"],dtype=np.float64)

    table_list:list[dict] = []
    for beta,a_wind in tqdm(product(beta_arr,a_wind_arr),total=beta_arr.size*a_wind_arr.size,desc="table"):
        table_row = _table_row(beta,a_wind,config_data["fixed"],config_data["units"])
        table_list.append(table_row)

    return pd.DataFrame(table_list)
