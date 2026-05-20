# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false
import numpy as np
import astropy.units as u
from astropy.constants import m_p

def calculate_rho_wind(
    a_wind:u.Quantity, 
    r:u.Quantity
)->u.Quantity:
    return (a_wind/(4.0*np.pi*r**2)).to(u.g/u.cm**3)

def calculate_n_wind(
    a_wind:u.Quantity,
    r:u.Quantity,
    mu:float,
)->u.Quantity:
    rho_wind = calculate_rho_wind(a_wind,r)
    return (rho_wind/(mu*m_p)).to(1/u.cm**3)
