# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false
import astropy.units as u
import numpy as np
from numpy.typing import NDArray
from module.tabular import ThermalSynchrotronTable
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

def calculate_P0(b_mag:u.Quantity):
    return (np.sqrt(3)*e**3*b_mag/(m_e*c**2)).to(u.erg/u.s/u.Hz)

def calculate_j0(
    nu_B:u.Quantity,
    n_ele:u.Quantity,
)->u.Quantity:
    quantity = (n_ele*e**2*nu_B/(np.sqrt(3.0)*c)).to(unit.emissivity)
    return quantity

def calculate_alpha0(
    nu_B:u.Quantity,
    n_ele:u.Quantity,
    pnu0:u.Quantity
)->u.Quantity:
    quantity = (n_ele*pnu0/(8.0*m_e*nu_B**2)).to(unit.absorption_coefficient)
    value = np.clip(quantity.value,1.0e-300,None)
    return value * quantity.unit

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
    theta:float,
    eps_B:float,
    a_wind:u.Quantity
)->u.Quantity:
    nu_t = (
        9.0 * theta**2 * e * np.sqrt(eps_B * a_wind)
    )/(
            8.0 * np.pi * m_e * c
        )
    return nu_t.to(u.s*u.GHz)

# def calculate_tau_theta(
#     eps_B: float,
#     theta: float,
#     mu: float,
#     mu_e: float,
#     beta_sh: float,
#     a_wind: u.Quantity
# ):
#     tau = (
#         2.0*e*mu_e/(3**(2.5)*theta**5*mu*m_p*c*beta_sh)
#             *np.sqrt(a_wind/eps_B)
#     )
#     return tau.to(u.dimensionless_unscaled)

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

# def calculate_tau_theta(
#     alpha_theta:u.Quantity,
#     r_theta:u.Quantity,
# )->np.ndarray:
#     tau_theta = np.atleast_1d((alpha_theta*r_theta).to_value(u.dimensionless_unscaled))
#     return tau_theta
#
def calculate_tau_theta(
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
    r_theta:u.Quantity,
    j_theta:u.Quantity,
    alpha_theta:u.Quantity
)->u.Quantity:
    l_theta = 4.0 * np.pi**2 * r_theta**2 * j_theta / alpha_theta
    return l_theta.to(unit.specific_luminosity)

def calculate_ln_tau(
    xi:NDArray[np.float64],
    tau_theta:NDArray[np.float64],
    table:ThermalSynchrotronTable
)->NDArray[np.float64]:
    log_ip_xi = table.calculate_log_ip(xi)
    return np.log(tau_theta) - np.log(xi) + log_ip_xi

def calculate_lnu_xi_dimless_tabular(
    xi:NDArray[np.float64],
    tau_theta:NDArray[np.float64],
    table:ThermalSynchrotronTable
)->NDArray[np.float64]:
    log_ip_xi = table.calculate_log_ip(xi)
    ip_xi = np.exp(log_ip_xi)
    lnu_xi = xi**2 * (-np.expm1(-tau_theta*ip_xi/xi))
    return lnu_xi
