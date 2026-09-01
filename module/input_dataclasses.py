from dataclasses import dataclass
from typing import Literal
from pathlib import Path

type ValueScale = Literal["linear","log"]
from module.types import FloatArray
from module.mydataclasses import (
    QuantityArray,
    QuantityArrayBase,
    QuantityData,
    YAMLReadable,
    ValueArray
)

@dataclass(frozen=True,slots=True)
class PhysicalParametersBase:
    a_wind: QuantityArrayBase
    beta_sh: ValueArray
    eps_b: float
    eps_th: float
    mu: float
    mu_e: float
    distance: QuantityData

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

