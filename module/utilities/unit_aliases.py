
""" Resigter aliases of physical unit

"""

# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false

import astropy.units as u
dimensionless = u.dimensionless_unscaled
emissivity = u.def_unit("erg/s/cm3/Hz",u.erg/u.s/u.cm**3/u.Hz)
number_density  = u.def_unit("/cm3",u.cm**(-3))
mass_density  = u.def_unit("g/cm3",u.g*u.cm**(-3))
enegy_density   = u.def_unit("erg/cm3",u.erg*u.cm**(-3))
magnetic_field = u.def_unit("G", enegy_density**(0.5))
wavenumber = u.def_unit("/cm",u.cm**(-1))
absorption_coefficient = u.def_unit("/cm",u.cm**(-1))
specific_luminosity = u.def_unit("erg/s/Hz", u.erg/u.s/u.Hz)
angular_frequency = u.def_unit("Hz rad", u.Hz*u.rad)
