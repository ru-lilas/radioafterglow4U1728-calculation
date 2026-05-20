""" electron_density.py
    Calculate 

"""
# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false

import astropy.units as u
from module.utilities import unit_aliases as unit

def calculate_n_ele_for_strong_shock(
    n_upstream:u.Quantity,
    mu_e:float
)->u.Quantity:
    return u.Quantity(
        4.0*n_upstream*mu_e,
        unit.number_density
    )

