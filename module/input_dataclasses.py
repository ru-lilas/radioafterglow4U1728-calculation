from dataclasses import dataclass
from typing import Literal

type ValueScale = Literal["linear","log"]

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
class FreeParameter:
    value_arr: VALUE_ARR
    scale: ValueScale
    unit: str|None

@dataclass
class FreeParameters:
    beta_sh: FreeParameter
    a_wind: FreeParameter

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
