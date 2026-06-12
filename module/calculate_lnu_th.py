import numpy as np
from numpy.typing import NDArray
from module.tabular import ThermalSynchrotronTable

def dimless_tabular_for_tau_theta(
    x:NDArray[np.float64],
    tau_theta:NDArray[np.float64],
    table:ThermalSynchrotronTable
)->NDArray[np.float64]:
    log_ip_xi = table.calculate_log_ip(x)
    ip_xi = np.exp(log_ip_xi)
    f_esc = -np.expm1(-tau_theta*ip_xi/x)
    return x**2 * f_esc

def dimless_tabular(
    x:NDArray[np.float64],
    log_tau:NDArray[np.float64],
)->NDArray[np.float64]:
    tau = np.exp(log_tau)
    f_esc = -np.expm1(-tau)
    return x**2 * f_esc
