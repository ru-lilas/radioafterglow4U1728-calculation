# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false

import astropy.units as u
from astropy.constants import m_p,c
import numpy as np
from module.utilities import unit_aliases as unit

def magnetic_field(
    n_us: u.Quantity[unit.number_density],
    beta_sh:float,
    eps_B: float,
    mu: float
)->u.Quantity[unit.magnetic_field]:
    return u.Quantity(
        np.sqrt(
            9.0*np.pi* eps_B*n_us*
            mu*m_p*(beta_sh*c)**2
        ),unit.magnetic_field
    )

def wind_profile(
    eps_B:float,
    a_wind:u.Quantity[u.g/u.cm],
    t:u.Quantity[u.s]
)->u.Quantity:
    return (1.5*np.sqrt(eps_B*a_wind)/t).to(unit.magnetic_field)
