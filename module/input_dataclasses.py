from dataclasses import dataclass

@dataclass
class MICROPHYSICS:
    eps_th: float
    eps_b: float
    mu: float
    mu_e: float

@dataclass
class VALUE_ARR:
    start: float
    stop: float
    num: int

@dataclass
class UNITS:
    phi_unit: str
    lnu_unit: str
    fnu_unit: str
    a_wind_unit: str

@dataclass
class DISTANCE:
    value: float
    unit: str

@dataclass
class SAMPLING:
    min: float
    max: float
    unit: str

@dataclass
class PhysicalParameters:
    beta_sh_arr: VALUE_ARR
    a_wind_arr: VALUE_ARR
    microphysics: MICROPHYSICS
    distance: DISTANCE
    units: UNITS
