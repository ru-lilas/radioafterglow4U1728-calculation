# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false
import astropy.units as u
import numpy as np
from astropy.constants import c
from numpy.typing import NDArray
from module.mydataclasses import QuantityData,QuantityArray
from module.types import FloatArray, FloatArrayLike

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
    beta: FloatArrayLike
):
    beta_arr = np.asarray(beta,dtype=np.float64)
    return u.Quantity(beta*c)

def beta_into_lorentz_gamma(
    beta: NDArray[np.float64]
):
    return 1.0/(np.sqrt(1.0 - beta**2))

def beta_into_doppler_delta(
    beta: FloatArrayLike
)->FloatArray:
    """
        真正面からの寄与以外を無視した場合の近似式
    """
    beta_arr = np.asarray(beta,dtype=np.float64)
    return np.sqrt((1.0+beta)/(1.0-beta))
