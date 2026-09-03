from dataclasses import dataclass
import numpy as np
from pathlib import Path
import pandas as pd
from .types import ValueScale,FloatArray
import astropy.units as u
from module.utils import FileReader
from cattrs import Converter
from typing import Self
DATACLASS_CONVERTER = Converter()

class YAMLReadable:
    @classmethod
    def from_yaml(
        cls,
        path:Path
    )->Self:
        dict_data = FileReader.yaml_safe(path)
        return DATACLASS_CONVERTER.structure(
            dict_data,
            cls
        )

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
class QuantityArrayBase:
    values: ValueArray
    unit: str

    @property
    def quantity(self):
        return u.Quantity(self.values.arr,self.unit)

    @property
    def quantity_array(self):
        return QuantityArray(
            values = self.values.arr,
            unit = self.unit
        )

    def FloatArray_in(self,unit:str|u.UnitBase)->FloatArray:
        value:FloatArray = np.asarray(self.quantity.to_value(unit),dtype=np.float64)
        return value

@dataclass(frozen=True,slots=True)
class QuantityArray:
    values: FloatArray
    unit: str

    @property
    def quantity(self):
        return u.Quantity(self.values,self.unit)

    def FloatArray_in(self,unit:str|u.UnitBase)->FloatArray:
        value:FloatArray = np.asarray(self.quantity.to_value(unit),dtype=np.float64)
        return value

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
