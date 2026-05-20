# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false
import astropy.units as u
import astropy.constants as const

def r_rad(
    t:u.Quantity,
    beta_sh:float
)->u.Quantity:
    return (t*beta_sh*const.c).to(u.cm)
