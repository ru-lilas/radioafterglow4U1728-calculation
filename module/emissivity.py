import numpy as np
from scipy import special
from module.synchrotron_function import thermal_Ip

def convert_xm(theta:float,chi:np.ndarray)->np.ndarray:
    return chi/(1.5*theta**2)

def j_th(
    theta:float,chi:np.ndarray
):
    x_theta = convert_xm(theta,chi)
    return chi/special.kv(2,1.0/theta)*thermal_Ip(x_theta)
