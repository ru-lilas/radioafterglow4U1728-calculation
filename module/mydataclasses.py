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

@dataclass(frozen=True, slots=True)
class LnSpace(YAMLReadable):
    start_log10: float
    stop_log10: float
    num: int

    def to_log10(self) -> NDArray[np.float64]:
        return np.linspace(
            self.start_log10,
            self.stop_log10,
            self.num,
            dtype=np.float64,
        )

    def to_ln(self) -> NDArray[np.float64]:
        return self.to_log10() * np.log(10.0)

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

    def inverted(self)->Self:
        return type(self)(
            values=np.logical_not(self.values),
        )

    def __invert__(self)->Self:
        return self.inverted()

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

    def find_nonmonotonic_lines(
        self,
        *,
        axis: int,
        rtol: float = 1.0e-10,
        atol: float = 0.0,
    ) -> NDArray[np.intp]:
        values = self.values

        if axis not in (0, 1):
            raise ValueError(
                f"axisは0または1である必要があります: axis={axis}"
            )

        diff = np.diff(values, axis=axis)

        if axis == 0:
            left = values[:-1, :]
            right = values[1:, :]
        else:
            left = values[:, :-1]
            right = values[:, 1:]

        scale = np.maximum(
            np.abs(left),
            np.abs(right),
        )
        tolerance = atol + rtol * scale

        increasing = (diff > tolerance).any(axis=axis)
        decreasing = (diff < -tolerance).any(axis=axis)

        nonmonotonic = increasing & decreasing

        return np.flatnonzero(nonmonotonic)

    def find_turning_points_in_columns(
        self,
        *,
        rtol: float = 1.0e-10,
        atol: float = 0.0,
    ) -> list[tuple[int, int]]:
        values = self.values
        turning_points: list[tuple[int, int]] = []

        for column in range(values.shape[1]):
            line = values[:, column]
            diff = np.diff(line)

            scale = np.maximum(
                np.abs(line[:-1]),
                np.abs(line[1:]),
            )
            tolerance = atol + rtol * scale

            # 数値誤差程度の差分を除外する
            significant_indices = np.flatnonzero(
                np.abs(diff) > tolerance
            )

            if significant_indices.size < 2:
                continue

            significant_signs = np.sign(
                diff[significant_indices]
            )

            sign_changed = (
                significant_signs[1:]
                != significant_signs[:-1]
            )

            # diff[i-1]とdiff[i]の符号が変わる場合、
            # values[i]が折り返し点に対応する
            turning_rows = significant_indices[1:][
                sign_changed
            ]

            for row in turning_rows:
                turning_points.append(
                    (int(row), column)
                )

        return turning_points

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

    def to_FloatGrid(
        self,
        unit:str
    )->FloatGrid:
        return FloatGrid(self.to_value_in(unit))




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

@dataclass(frozen=True, slots=True)
class MyVector:
    values: NDArray
    def __post_init__(self) -> None:
        if self.values.ndim != 1:
            raise ValueError(
                "valuesは1次元配列でなければなりません. "
                f"shape={self.values.shape}"
            )

@dataclass(frozen=True, slots=True)
class BoolVector(MyVector):
    pass

@dataclass(frozen=True, slots=True)
class FloatVector(MyVector):
    values: NDArray[np.float64]| np.ma.MaskedArray

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

@dataclass(frozen=True, slots=True)
class NDArraySpace:
    start: float
    stop: float
    num: int

    def linspace(self) -> NDArray[np.float64]:
        return np.linspace(
            start=self.start,
            stop=self.stop,
            num=self.num,
        )

    def geomspace(self) -> NDArray[np.float64]:
        if self.start <= 0.0 or self.stop <= 0.0:
            raise ValueError(
                "対数間隔の配列生成には"
                "正の値が必要です"
            )
        return np.geomspace(
            start=self.start,
            stop=self.stop,
            num=self.num,
        )
