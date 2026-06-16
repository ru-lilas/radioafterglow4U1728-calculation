from typing import Any
import numpy as np
from numpy.typing import NDArray
from module.find_peak_properties import build_peak_property
from module.models import InputParameters
from module.tabular import ThermalSynchrotronTable
import pandas as pd
from dataclasses import asdict
from tqdm import tqdm

def build_xm_arr(
    input_conf:dict,
    tabular:ThermalSynchrotronTable
)->NDArray[np.float64]:
    xm_num:int = input_conf["xm_num"]
    return np.logspace(
        start=np.log(tabular.xi_min),
        stop=np.log(tabular.xi_max),
        num=xm_num,
        dtype=np.float64
    )

def build_tau_theta_arr(
    input_conf:dict[str,Any],
)->NDArray[np.float64]:
    return np.logspace(
        **input_conf["tau_theta_arr"],
        dtype=np.float64
    )

def peak_table(
    xm_arr:NDArray[np.float64],
    tau_theta_arr:NDArray[np.float64],
    integral_table:ThermalSynchrotronTable,
):
    peak_data = build_peak_property(xm_arr,tau_theta_arr,integral_table)
    return peak_data

def varying(
    xm_arr:NDArray[np.float64],
    tabular:ThermalSynchrotronTable,
    inputparams:list[InputParameters]
)->tuple[dict[str,Any],pd.DataFrame]:
    outdata:list[dict] = []
    for inputparam in tqdm(inputparams):
        tau_theta = inputparam.tau_theta
        peak_data = build_peak_property(xm_arr,tau_theta,tabular)
        outdata.append({
            **peak_data,
            **asdict(inputparam),
            "phi_theta": inputparam.phi_theta.value,
            "phi_theta_unit": inputparam.phi_theta.unit,
            "l_theta": inputparam.l_theta.value,
            "l_unit": inputparam.l_theta.unit,
        })
    df = pd.DataFrame(outdata)
    return {},df
