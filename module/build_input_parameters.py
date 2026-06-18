import pandas as pd
from typing import Any
from numpy.typing import NDArray
from module.models import InputParameters
import numpy as np

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
) -> dict[str,Any]:

    params = InputParameters(
        **fixed,
        beta_sh=beta_sh,
        a_wind_value=a_wind_value,
    )

    return {
        "beta_sh": beta_sh,
        "a_wind_value": a_wind_value,
        "phi_theta": params.phi_theta.value,
        "phi_unit": params.phi_theta.unit,
        "l_theta": params.l_theta.value,
        "l_unit": params.l_theta.unit,
        "tau_theta": params.tau_theta,
    }

def table(
    input:dict[str,Any]
)->pd.DataFrame:
    a_wind_arr:NDArray[np.float64] = \
        np.logspace(**input["a_wind_arr"],dtype=np.float64)
    beta_arr:NDArray[np.float64] = \
        np.logspace(**input["beta_arr"],dtype=np.float64)

    table_list:list[dict] = []
    for beta in beta_arr:
        for a_wind in a_wind_arr:
            table_row = _table_row(beta,a_wind,input["fixed"])
            table_list.append(table_row)

    return pd.DataFrame(table_list)

