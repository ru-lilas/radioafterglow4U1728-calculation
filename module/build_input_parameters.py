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
