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

def beta_into_lorentz_gamma(
    beta: NDArray[np.float64]
):
    return 1.0/(np.sqrt(1.0 - beta**2))

def beta_into_doppler_delta(
    beta: float
)->float:
    """
        真正面からの寄与以外を無視した場合の近似式
    """
    return np.sqrt((1.0+beta)/(1.0-beta))
