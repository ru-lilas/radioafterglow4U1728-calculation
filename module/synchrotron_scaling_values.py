# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false
import astropy.units as u
import numpy as np
from module.utilities import unit_aliases as unit
from astropy.constants import e,c,m_e
e = u.Quantity(e.esu)

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
