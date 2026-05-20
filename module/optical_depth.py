# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false
import numpy as np
import astropy.units as u
from module.utilities import unit_aliases as unit

def calculate_tau_nu(
    alpha_nu:u.Quantity,
    r:u.Quantity
)->np.ndarray:
    return (r*alpha_nu).to(unit.dimensionless)

def calculate_escape_fraction(
    tau_nu: np.ndarray
)->np.ndarray:
    return -np.expm1(-tau_nu)

