# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false

import astropy.units as u
from scipy import special
from module.utilities import unit_aliases as unit

def exact(theta:float)->float:
    return (
        2.0*theta**2 / special.kv(2,1.0/theta)
    )

def hotlimit():
    return u.Quantity(1.0,unit.dimensionless)
