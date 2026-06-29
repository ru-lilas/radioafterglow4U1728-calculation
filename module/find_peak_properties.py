import numpy as np
from numpy.typing import NDArray
from module import calculate_lnu_th
from module.tabular import ThermalSynchrotronTable

def build_peak_property(
    xm_arr:NDArray[np.float64],
    tau_theta:NDArray[np.float64],
    tabular: ThermalSynchrotronTable
):
    lnu_th_dimless = calculate_lnu_th.dimless_tabular_for_tau_theta(
        x=xm_arr,
        tau_theta=tau_theta,
        table=tabular
    )

    idx_peak: int = int(np.argmax(lnu_th_dimless))
    lnu_peak_dimless:np.float64 = lnu_th_dimless[idx_peak]
    xm_peak:np.float64 = xm_arr[idx_peak]

    return{
        "tau_theta": tau_theta,
        "xm_peak": xm_peak,
        "lambda_peak": lnu_peak_dimless
    }
