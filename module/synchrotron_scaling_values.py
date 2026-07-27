# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false
import astropy.units as u
import numpy as np
from numpy.typing import NDArray
from module.tabular import ThermalSynchrotronTable
from module.types import FloatArray, FloatArrayLike
from module.utilities import unit_aliases as unit
from astropy.constants import e,c,m_e,m_p
from module import f_factor
e = u.Quantity(e.esu)

def calculate_t_peak_nu(
    eps_B: float,
    theta: float,
    a_wind: u.Quantity,
    nu: u.Quantity,
):
    t_peak = (
        9.0 * theta**2 * e * np.sqrt(eps_B * a_wind)
    )/(
            8.0 * np.pi * m_e * c * nu
        )
    return t_peak.to(u.s) 

def convert_nu_into_omega(nu:u.Quantity)->u.Quantity:
    return nu/(2.0*np.pi)

def convert_omega_into_nu(omega:u.Quantity)->u.Quantity:
    return omega*2.0*np.pi

def omega_b(
    b_mag:u.Quantity
)->u.Quantity:
    """
        calculate gyro frequency for given B
    """
    omega_b = (e*b_mag/(m_e*c)).to(u.Hz)
    omega_b_val = np.clip(omega_b.value, 1e-300, None) # avoid devision by zero
    return omega_b_val*omega_b.unit

def chi_gyro(
    omega:u.Quantity[unit.angular_frequency],
    omega_b:u.Quantity[unit.angular_frequency]
)->np.ndarray:
    """
        calculate frequency devided by gyro frequency
    """
    value = (omega/omega_b).to_value(unit.dimensionless)
    value = np.clip(omega/omega_b,1e-300,None)
    return value

def calculate_nu_crit(
    nu_B:u.Quantity,
    theta:float
)->u.Quantity:
    return (1.5 * theta**2 * nu_B).to(u.Hz)

def calculate_nu_b(
        b_mag:u.Quantity
)->u.Quantity:
    return convert_omega_into_nu(omega_b(b_mag))

def calculate_phi_theta(
    theta:FloatArrayLike,
    eps_b:FloatArrayLike,
    a_wind:u.Quantity
)->u.Quantity:
    theta_arr = np.asarray(theta,dtype=np.float64)
    eps_b_arr = np.asarray(eps_b,dtype=np.float64)
    nu_t = (
        9.0 * theta**2 * e * np.sqrt(eps_b * a_wind)
    )/(
            8.0 * np.pi * m_e * c
        )
    return nu_t

def calculate_j_theta(
        theta:float,
        n_ele_theta:u.Quantity,
        b_mag_theta:u.Quantity
)->u.Quantity:
    f_theta = f_factor.exact(theta)
    j_theta = (
            np.sqrt(3)*e**3*n_ele_theta*b_mag_theta*f_theta
        )/(
            8.0*np.pi*m_e*c**2
        )
    return j_theta.to(unit.emissivity)

def calculate_alpha_theta(
    theta:float,
    n_ele_theta:u.Quantity,
    b_mag_theta:u.Quantity
)->u.Quantity:
    f_theta = f_factor.exact(theta)
    alpha_theta = (
            np.pi*e*n_ele_theta*f_theta
    )/(
            3**(1.5)*theta**5*b_mag_theta
        )
    return alpha_theta.to(unit.absorption_coefficient)

def calculate_tau_theta(
    eps_b: FloatArrayLike,
    mu_e: FloatArrayLike,
    mu: FloatArrayLike,
    theta: FloatArrayLike,
    a_wind: u.Quantity,
    beta_sh: FloatArrayLike
)->FloatArray:
    eps_b_arr = np.asarray(eps_b,dtype=np.float64)
    mu_e_arr = np.asarray(mu_e,dtype=np.float64)
    mu_arr = np.asarray(mu,dtype=np.float64)
    theta_arr = np.asarray(theta,dtype=np.float64)
    beta_sh_arr = np.asarray(beta_sh,dtype=np.float64)
    quantities = (e*np.sqrt(a_wind)/(m_p*c)).decompose()
    dimless_term = 2.0*mu_e_arr/ (3**(2.5)*theta_arr**5*mu_arr*beta_sh_arr*np.sqrt(eps_b_arr))
    return np.asarray(dimless_term*quantities.to_value(unit.dimensionless),dtype=np.float64)

def calculate_tau_theta_ar(
    alpha_theta: u.Quantity,
    r_theta: u.Quantity,
)->NDArray[np.float64]:
    tau_theta = (alpha_theta * r_theta).to_value(
        u.dimensionless_unscaled
    )
    return np.atleast_1d(np.asarray(tau_theta,dtype=float))

def calculate_t_theta(phi_theta:u.Quantity,nu:u.Quantity)->u.Quantity:
    t_theta = phi_theta/nu
    return t_theta.to(u.s)

def calculate_r_theta(beta_sh:float,t_theta:u.Quantity)->u.Quantity:
    r_theta = beta_sh*c*t_theta
    return r_theta.to(u.cm)

def calculate_n_ele_theta(
    beta_sh:float,
    mu:float,
    mu_e:float,
    a_wind:u.Quantity,
    t_theta:u.Quantity
)->u.Quantity:
    # strong shock, compression ratio = 4
    n_ele_theta = mu_e*a_wind/(np.pi*mu*m_p*(beta_sh*c*t_theta)**2)
    return n_ele_theta.to(unit.number_density)

def calculate_b_mag_theta(
    eps_B:float,
    a_wind:u.Quantity[u.g/u.cm],
    t_theta:u.Quantity[u.s]
)->u.Quantity:
    return (1.5*np.sqrt(eps_B*a_wind)/t_theta).to(unit.magnetic_field)

def calculate_l_theta(
    beta_sh: FloatArrayLike,
    eps_b: FloatArrayLike,
    theta: FloatArrayLike,
    a_wind: u.Quantity
)->u.Quantity:
    beta_sh_arr = np.asarray(beta_sh,dtype=np.float64)
    eps_b_arr = np.asarray(eps_b,dtype=np.float64)
    theta_arr = np.asarray(theta,dtype=np.float64)
    return (81.0*beta_sh_arr**2*theta_arr**5*eps_b_arr*a_wind*e**2/(8.0*m_e))

def calculate_ln_tau(
    xi:NDArray[np.float64],
    tau_theta:NDArray[np.float64],
    table:ThermalSynchrotronTable
)->NDArray[np.float64]:
    log_ip_xi = table.calculate_log_ip(xi)
    return np.log(tau_theta) - np.log(xi) + log_ip_xi

def calculate_lambda_using_table(
    xm:NDArray[np.float64],
    tau_theta:NDArray[np.float64],
    table:ThermalSynchrotronTable
)->NDArray[np.float64]:
    log_ip_xi = table.calculate_log_ip(xm)
    ip_xi = np.exp(log_ip_xi)
    tauip = ip_xi*tau_theta
    f_esc = -np.expm1(-tauip/xm)
    xm2 = xm**2
    return xm2 * f_esc
