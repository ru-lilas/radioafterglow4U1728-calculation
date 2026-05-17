import numpy as np
from scipy import special,integrate
from module.synchrotron_function import thermal_Ip,angle_averaged_dimensionless_synchrotron_power as Pi_nu

def convert_xm(theta:float,chi:np.ndarray)->np.ndarray:
    return chi/(1.5*theta**2)

def j_th(
    theta:float,chi:np.ndarray
):
    x_theta = convert_xm(theta,chi)
    return chi/special.kv(2,1.0/theta)*thermal_Ip(x_theta)

def compute_anu_th_integral(
    theta: float,
    chi: np.ndarray,
    Nu: int = 200,
)->np.ndarray:
    # integral range
    u = np.logspace(-4, 3, Nu)

    # calculate Lorentz factor
    gamma = np.sqrt(1.0 + u**2)

    # <Pi_nu>(chi, u)
    Pi_val = Pi_nu(chi,u)

    # integrand
    integrand = (
        Pi_val
        * np.exp(-gamma / theta)
        * (1.0/gamma - u / theta)
    )

    # 積分
    integral = integrate.trapezoid(integrand, u, axis=-1)

    return integral

def anu_th_dimless(
    chi: np.ndarray,
    theta: float,
    Nu: int = 200,
) -> np.ndarray:

    integral = compute_anu_th_integral(theta=theta,chi=chi,Nu=Nu)

    # prefactor
    K2 = special.kv(2, 1.0/theta)

    value = - integral / (chi**2*theta * K2)
    value = np.clip(value,1e-100,None)
    return value
