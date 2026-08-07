from dataclasses import dataclass
import numpy as np
from pathlib import Path
import pandas as pd
from .types import ValueScale,FloatArray
import astropy.units as u
from module.utilities.filereaders import YAMLReadable

@dataclass(frozen=True,slots=True)
class ValueArray(YAMLReadable):
    start: float
    stop: float
    num: int
    scale: str

    @property
    def arr(self)->FloatArray:
        if (self.scale == "linear"):
            return np.linspace(start=self.start,stop=self.stop,num=self.num,dtype=np.float64)
        else:
            return np.logspace(start=self.start,stop=self.stop,num=self.num,dtype=np.float64)

@dataclass(frozen=True,slots=True)
class QuantityArray:
    values: ValueArray
    unit: str

    @property
    def quantity(self):
        return u.Quantity(self.values.arr,self.unit)

@dataclass(frozen=True,slots=True)
class QuantityData:
    value: float
    unit: str

    @property
    def quantity(self):
        return u.Quantity(self.value,self.unit)

    def value_in(self,unit:str)->float:
        value: float = float(np.asarray(self.quantity.to_value(unit)).item())
        return value
