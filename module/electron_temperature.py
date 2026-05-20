# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false

from astropy.constants import m_e,m_p
import numpy as np

def calculate_theta0(
    eps_th:float,
    mu:float,
    mu_e:float,
    beta_sh:float
)->float:
    return (
        9.0*mu*m_p*eps_th*beta_sh**2/(32.0*mu_e*m_e)
    )

def calculate_theta_gp98(
    theta0:float
)->float:
    return (
            (5.0*theta0 -6.0 +
                np.sqrt(
                    25.0*theta0**2
                    + 180.0*theta0
                    + 36.0
                )
            )
            /30.0)

def calculate_hotlimit(
    theta0:float
)->float:
        return theta0/3.0

def calculate_coldlimit(
    theta0:float
)->float:
    return (2.0*theta0/3.0)

def calculate_theta_e(
    eps_th:float,
    mu:float,
    mu_e:float,
    beta_sh:float
)-> float:
    theta0 = calculate_theta0(eps_th,mu,mu_e,beta_sh)
    return calculate_theta_gp98(theta0)
