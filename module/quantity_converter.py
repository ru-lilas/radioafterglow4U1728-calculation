# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false
import astropy.units as u
import numpy as np
from astropy.constants import c
from numpy.typing import NDArray

def flux_into_luminosity(
    flux:u.Quantity,
    distance:u.Quantity
):
    return u.Quantity(4.0*np.pi*distance**2*flux)

def lnu_into_fnu(
    lnu: u.Quantity,
    distance: u.Quantity
):
    return u.Quantity(lnu/(4.0*np.pi*distance**2))

def beta_into_velocity(
    beta: NDArray[np.float64]
):
    return u.Quantity(beta*c)
