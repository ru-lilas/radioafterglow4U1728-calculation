import astropy.units as u
import numpy as np

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
