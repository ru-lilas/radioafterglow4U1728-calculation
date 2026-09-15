from dataclasses import dataclass, replace
import numpy as np
from pathlib import Path
from numpy.typing import NDArray
import pandas as pd
from .types import ValueScale,FloatArray
import astropy.units as u
from module.utils import FileReader
from cattrs import Converter
from typing import Any, Self
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

    def to_floatarray(self)->FloatArray:
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

    def to_QuantityArray1D(self):
        return QuantityArray1D(
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

    @classmethod
    def from_Quantity(
        cls,
        quantity:Any
    )->Self:
        if not isinstance(quantity,u.Quantity):
            raise ValueError("エラー:引数がQuantityではありません.")
        values = np.asarray(quantity.value,dtype=np.float64)
        unit = quantity.unit
        if unit is None:
            raise ValueError("エラー:物理量に単位がありません.")
        else:
            return cls(
                values=values,
                unit=unit.to_string()
            )

    @property
    def quantity(self):
        return u.Quantity(self.values,self.unit)

    def to_value_in(self,unit:str)->FloatArray:
        return np.asarray(
            self.quantity.to_value(unit),
            dtype=np.float64
        )

@dataclass(frozen=True, slots=True)
class QuantityArray1D(QuantityArray):
    def __post_init__(self) -> None:
        if self.values.ndim != 1:
            raise ValueError(
                "valuesは1次元配列でなければなりません. "
                f"shape={self.values.shape}"
            )

@dataclass(frozen=True, slots=True)
class QuantityGrid(QuantityArray):
    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(
                "valuesは2次元配列でなければなりません. "
                f"shape={self.values.shape}"
            )
    @classmethod
    def from_df(
        cls,
        df:pd.DataFrame,
        column:str,
        unit:str,
        index_name:str = "index"
    )->Self:
        values = (
            df.sort_values(index_name)[column]
            .to_numpy(dtype=np.float64)
        )
        n = int(np.sqrt(values.size))

        if n * n != values.size:
            raise ValueError(
                f"Cannot reshape {values.size} elements into a square grid."
            )
        return cls(
            values=values.reshape(n,n),
            unit=unit
        )

@dataclass(frozen=True, slots=True)
class BoolGrid:
    values: NDArray[np.bool_]
    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(
                "valuesは2次元配列でなければなりません. "
                f"shape={self.values.shape}"
            )
    @classmethod
    def from_df(
        cls,
        df:pd.DataFrame,
        column:str,
        index_name:str = "index"
    )->Self:
        values = (
            df.sort_values(index_name)[column]
            .to_numpy(dtype=np.bool_)
        )
        n = int(np.sqrt(values.size))

        if n * n != values.size:
            raise ValueError(
                f"Cannot reshape {values.size} elements into a square grid."
            )
        return cls(
            values=values.reshape(n,n),
        )

@dataclass(frozen=True, slots=True)
class FloatGrid:
    values: NDArray[np.float64]| np.ma.MaskedArray
    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(
                "valuesは2次元配列でなければなりません. "
                f"shape={self.values.shape}"
            )

    @property
    def shape(self):
        return self.values.shape

    def masking(
        self,
        mask:BoolGrid,
        condition:bool = True
    )->Self:
        val = self.values
        if condition:
            return replace(
                self,
                values=np.ma.masked_where(
                    mask.values.reshape(val.shape),
                    val,
                )
            )
        else:
            return replace(
                self,
                values=np.ma.masked_where(
                    ~mask.values.reshape(val.shape),
                    val,
                )
            )



@dataclass(frozen=True,slots=True)
class QuantityData:
    value: float
    unit: str

    @classmethod
    def from_Quantity(
        cls,
        quantity:u.Quantity
    )->Self:
        value = float(quantity.value)
        unit = quantity.unit
        if unit is None:
            raise ValueError("エラー:物理量に単位がありません.")
        else:
            return cls(
                value=value,
                unit=unit.to_string()
            )

    @property
    def quantity(self):
        return u.Quantity(self.value,self.unit)

    def value_in(self,unit:str)->float:
        value: float = float(np.asarray(self.quantity.to_value(unit)).item())
        return value
