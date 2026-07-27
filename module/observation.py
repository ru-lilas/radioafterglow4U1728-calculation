from pathlib import Path
from dataclasses import dataclass
from typing import Self, cast
from dacite import from_dict
from numpy.typing import NDArray
from functools import cached_property
from module.models import ThermalSynchrotron, ThermalSynchrotronTable, calculate_phi, calculate_xm
from module.mydataclasses import QuantityData,QuantityArray
from module.types import FloatArray
from module.utilities import filereaders as fr
import astropy.units as u
import numpy as np
import pandas as pd

@dataclass
class LightcurveMetadata:
    t_bin: float
    t_unit: str
    fnu_unit: str
    nu_unit: str

    @classmethod
    def from_keyvalue(
            cls,
            path:Path
    )->Self:
        dict_data = fr.read_keyvalue(path)
        return cls(**dict_data)

    @cached_property
    def bin_width(self):
        return u.Quantity(self.t_bin,self.t_unit)

@dataclass
class Lightcurve:
    nu: float
    df: pd.DataFrame
