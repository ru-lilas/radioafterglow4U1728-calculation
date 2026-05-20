# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false
import numpy as np
import astropy.units as u
from module.utilities import unit_aliases as unit

def calculate_lnu_th(
    snu_th: u.Quantity,
    f_esc: np.ndarray,
    r: u.Quantity
)->u.Quantity:
    return (4.0 * np.pi**2 * r**2 * snu_th * f_esc).to(unit.specific_luminosity)

