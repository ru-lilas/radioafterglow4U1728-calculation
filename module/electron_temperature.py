# pyright: reportAttributeAccessIssue=false

from astropy.constants import m_e,m_p
import numpy as np

from module.types import FloatArray, FloatArrayLike

MASS_RATIO = float((m_p / m_e).decompose().value)

def as_float_array(value:FloatArrayLike)->FloatArray:
    return np.asarray(value,dtype=np.float64)

def calculate_theta0(
    eps_th: FloatArrayLike,
    mu: FloatArrayLike,
    mu_e: FloatArrayLike,
    beta_sh: FloatArrayLike
)->FloatArray:
    eps_th_arr = as_float_array(eps_th)
    mu_arr = as_float_array(mu)
    mu_e_arr = as_float_array(mu_e)
    beta_sh_arr = as_float_array(beta_sh)
    return np.asarray((
        9.0
        *mu_arr
        *MASS_RATIO
        *eps_th_arr
        *beta_sh_arr**2
        /(32.0*mu_e_arr)
    ),
        dtype=np.float64
    )

def calculate_theta_gp98(
    theta0: FloatArrayLike,
) -> FloatArray:
    theta0_arr = as_float_array(theta0)

    return np.asarray(
        (
            5.0 * theta0_arr
            - 6.0
            + np.sqrt(
                25.0 * theta0_arr**2
                + 180.0 * theta0_arr
                + 36.0
            )
        )
        / 30.0,
        dtype=np.float64,
    )

def calculate_hotlimit(
    theta0: FloatArrayLike,
) -> FloatArray:
    return as_float_array(theta0) / 3.0

def calculate_coldlimit(
    theta0: FloatArrayLike,
) -> FloatArray:
    return 2.0 * as_float_array(theta0) / 3.0

def calculate_theta_e(
    eps_th: FloatArrayLike,
    mu: FloatArrayLike,
    mu_e: FloatArrayLike,
    beta_sh: FloatArrayLike,
) -> FloatArray:
    theta0 = calculate_theta0(
        eps_th=eps_th,
        mu=mu,
        mu_e=mu_e,
        beta_sh=beta_sh,
    )

    return calculate_theta_gp98(theta0)
#
# def calculate_theta_gp98(
#     theta0:float
# )->float:
#     return (
#             (5.0*theta0 -6.0 +
#                 np.sqrt(
#                     25.0*theta0**2
#                     + 180.0*theta0
#                     + 36.0
#                 )
#             )
#             /30.0)
#
# def calculate_hotlimit(
#     theta0:float
# )->float:
#         return theta0/3.0
#
# def calculate_coldlimit(
#     theta0:float
# )->float:
#     return (2.0*theta0/3.0)
#
# def calculate_theta_e(
#     eps_th:float,
#     mu:float,
#     mu_e:float,
#     beta_sh:float
# )-> float:
#     theta0 = calculate_theta0(eps_th,mu,mu_e,beta_sh)
#     return calculate_theta_gp98(theta0)
