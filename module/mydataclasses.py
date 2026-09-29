from dataclasses import dataclass, replace
from os import wait
import numpy as np
from pathlib import Path
from numpy.typing import NDArray
import pandas as pd
from .types import FloatArray1D, ValueScale,FloatArray
import astropy.units as u
from module.utils import FileReader
from cattrs import Converter
from typing import Any, Literal, Self
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

    def to_floatvector(self):
        qarr = self.to_QuantityArray1D()
        return FloatVector(
            values = qarr.quantity.decompose()
        )

    def FloatArray_in(self,unit:str|u.UnitBase)->FloatArray:
        value:FloatArray = np.asarray(self.quantity.to_value(unit),dtype=np.float64)
        return value

@dataclass(frozen=True,slots=True)
class QuantityArray:
    values: FloatArray
    unit: str

    @classmethod
    def from_quantity(
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

    @property
    def astropy_unit(self)->u.UnitBase:
        unit = self.quantity.unit
        if isinstance(unit,u.UnitBase):
            return unit
        else:
            raise TypeError("u.Unitまたはu.UnitBaseへ変換できません.")


@dataclass(frozen=True, slots=True)
class VectorBase:
    values:NDArray
    def __post_init__(self) -> None:
        if self.values.ndim != 1:
            raise ValueError(
                "valuesは1次元配列でなければなりません. "
                f"shape={self.values.shape}"
            )
            
@dataclass(frozen=True, slots=True)
class IntVector(VectorBase):
    values:NDArray[np.int64]

    @classmethod
    def from_df(
        cls,
        df:pd.DataFrame,
        index_name:str
    )->Self:
        return cls(
            values = np.asarray(df[index_name],dtype=np.int64)
        )

    def as_scalar(
        self
    )->int:
        values = self.values

        if values.size == 0:
            raise ValueError("values is empty.")

        if not np.all(values == values[0]):
            raise ValueError("All values must be equal.")

        return int(values[0])

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
    @classmethod
    def from_series(
        cls,
        row:pd.Series,
        *,
        column:str,
        unit:str
    )->Self:
        value = row[column]
        if not isinstance(value,float):
            raise TypeError(f"エラー:無効なvalueです:{value}")
        else:
            return cls(
                value=value,
                unit=unit
            )
    @classmethod
    def from_attrs(
        cls,
        df:pd.DataFrame,
        *,
        key:str,
        unit_key:str|None = None
    )->Self:
        value = df.attrs[key]
        if not isinstance(value,float):
            raise TypeError("エラー:無効なvalueです.")
        if not unit_key is None:
            unit = df.attrs[unit_key]
            if not isinstance(unit,str):
                raise TypeError("エラー:無効なunitです.")
        else:
            unit = ""

        return cls(
            value=value,
            unit=unit
        )


    @property
    def quantity(self):
        return u.Quantity(self.value,self.unit)

    def value_in(self,unit:str)->float:
        value: float = float(np.asarray(self.quantity.to_value(unit)).item())
        return value

@dataclass(frozen=True, slots=True)
class QuantityArray1D(QuantityArray):
    def __post_init__(self) -> None:
        if self.values.ndim != 1:
            raise ValueError(
                "valuesは1次元配列でなければなりません. "
                f"shape={self.values.shape}"
            )

    @classmethod
    def from_df(
        cls,
        df:pd.DataFrame,
        column:str,
        unit_key:str,
    )->Self:
        return cls(
            values=df[column].to_numpy(dtype=np.float64),
            unit=df.attrs[unit_key]
        )

    @classmethod
    def from_df_unit(
        cls,
        df:pd.DataFrame,
        column:str,
        unit:str,
    )->Self:
        return cls(
            values=df[column].to_numpy(dtype=np.float64),
            unit=unit
        )

    def decompose(
        self,
    )->FloatArray1D:
        return FloatArray1D(
            values = self.quantity.decompose()
        )

    def index(
        self,
        index:int
    )->QuantityData:
        return QuantityData(
            value=self.values[index],
            unit=self.unit
        )

    @property
    def length(self):
        return self.values.size

    def to_floatarray1d(
        self,
        unit: str
    ):
        return FloatArray1D(self.to_value_in(unit))



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

    def take_and(
        self,
        boolgrids: list[Self],
    ) -> Self:
        grids = [self, *boolgrids]

        if any(
            grid.values.shape != self.values.shape
            for grid in grids
        ):
            shapes = [
                grid.values.shape
                for grid in grids
            ]
            raise ValueError(
                "すべてのBoolGridのshapeが"
                "一致している必要があります: "
                f"shapes={shapes}"
            )

        result = np.logical_and.reduce(
            [grid.values for grid in grids]
        )

        return type(self)(
            values=np.asarray(
                result,
                dtype=np.bool_,
            )
        )
    def take_or(
        self,
        boolgrids: list[Self],
    ) -> Self:
        grids = [self, *boolgrids]

        if any(
            grid.values.shape != self.values.shape
            for grid in grids
        ):
            shapes = [
                grid.values.shape
                for grid in grids
            ]
            raise ValueError(
                "すべてのBoolGridのshapeが"
                "一致している必要があります: "
                f"shapes={shapes}"
            )

        result = np.logical_or.reduce(
            [grid.values for grid in grids]
        )

        return type(self)(
            values=np.asarray(
                result,
                dtype=np.bool_,
            )
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

    @classmethod
    def from_1DArrays(
        cls,
        row: FloatArray1D,
        column: FloatArray1D,
    )->Self:
        return cls(
            values = np.outer(
                row.values,column.values
            ),
        )

    @property
    def shape(self):
        return self.values.shape

    @property
    def ln(self):
        return type(self)(values=np.log(self.values))

    @property
    def log10(self):
        return type(self)(values=np.log10(self.values))

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

    def nearest_values(
        self,
        value_crit: float
    )->BoolGrid:
        values = self.values   
        row_indices = np.argmin(
            np.abs(values - value_crit),
            axis=0
        )
        column_indices = np.arange(values.shape[1])
        mask = np.zeros_like(
            values,
            dtype=bool,
        )
        mask[row_indices, column_indices] = True
        return BoolGrid(mask)

    def less_than(
        self,
        value_crit: float
    )->BoolGrid:
        return BoolGrid(
            values=self.values < value_crit
        )

    def greater_than(
        self,
        value_crit: float
    )->BoolGrid:
        return BoolGrid(
            values=self.values > value_crit
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
class GridRange:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

@dataclass(frozen=True, slots=True)
class FloatGridPair:
    x: FloatGrid
    y: FloatGrid

    def __post_init__(self):
        if self.x.shape != self.y.shape:
            raise ValueError(
                "xとyのグリッドshapeが一致していません: "
                f"x={self.x.shape}, y={self.y.shape}"
            )

    def is_contained_in(
        self,
        gridrange: GridRange,
    ) -> BoolGrid:
        x = self.x.values
        y = self.y.values

        contained = (
            (gridrange.xmin <= x)
            & (x <= gridrange.xmax)
            & (gridrange.ymin <= y)
            & (y <= gridrange.ymax)
        )

        return BoolGrid(
            values=np.asarray(
                np.ma.filled(contained, False),
                dtype=np.bool_,
            )
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
    def from_1DArrays(
        cls,
        row: QuantityArray1D,
        column: QuantityArray1D,
    )->Self:
        return cls(
            values = np.outer(
                row.values,column.values
            ),
            unit = (row.astropy_unit*column.astropy_unit).to_string()
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

    @classmethod
    def from_FloatGrid(
        cls,
        grid: FloatGrid,
        row_quantity: QuantityArray1D
    ):
        return cls.from_quantity(
            grid.values*row_quantity.quantity[:,None]
        )

    def to_FloatGrid(
        self,
        unit:str
    )->FloatGrid:
        return FloatGrid(self.to_value_in(unit))

    def divide_by_column(
        self,
        column: QuantityArray1D,
        *,
        decompose: bool
    ) -> Self|FloatGrid:
        quantity = self.quantity/self.quantity[:,None]
        if decompose:
            return FloatGrid(
                values = quantity.decompose()
            )
        else:
            return type(self)(
                values = quantity.value,
                unit = (
                    self.astropy_unit/column.astropy_unit
                ).to_string()
            )
    def decompose_by_column(
        self,
        column: QuantityArray1D,
    ) -> FloatGrid:
        quantity = self.quantity/column.quantity[:,None]
        return FloatGrid(
            values = np.asarray(quantity.decompose().value,dtype=np.float64)
        )



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

@dataclass(frozen=True, slots=True)
class FloatArray1DSpace(NDArraySpace):
    scale: ValueScale

    def to_FloatArray1D(self):
        if self.scale=="linear":
            return FloatArray1D(self.linspace())
        elif self.scale=="log":
            return FloatArray1D(self.geomspace())
        else:
            raise ValueError(
                f"未対応のscaleです:{self.scale}"
            )

@dataclass(frozen=True, slots=True)
class FloatVectorSpace(NDArraySpace):
    scale: ValueScale

    def to_floatvector(self):
        if self.scale=="linear":
            return FloatVector(self.linspace())
        elif self.scale=="log":
            return FloatVector(self.geomspace())
        else:
            raise ValueError(
                f"未対応のscaleです:{self.scale}"
            )


@dataclass(frozen=True, slots=True)
class QuantityArray1DSpace:
    values: FloatArray1DSpace
    unit: str

    def to_QuantityArray1D(self):
        return QuantityArray1D(
            values = self.values.to_FloatArray1D().values,
            unit = self.unit
        )

@dataclass(frozen=True, slots=True)
class IndexGrid:
    values: NDArray[np.int64]

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(
                "valuesは2次元配列でなければなりません: "
                f"shape={self.values.shape}"
            )

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
    ) -> Self:
        values = df.index.to_numpy(dtype=np.int64)

        n = int(np.sqrt(values.size))

        if n * n != values.size:
            raise ValueError(
                f"{values.size}個のindexを"
                "正方形gridへ変換できません"
            )

        return cls(
            values=values.reshape(n, n),
        )

    def select(
        self,
        mask: BoolGrid,
    ) -> NDArray[np.int64]:
        if self.shape != mask.values.shape:
            raise ValueError(
                "indexとmaskのshapeが一致していません: "
                f"index={self.shape}, "
                f"mask={mask.values.shape}"
            )

        return np.asarray(
            self.values[mask.values],
            dtype=np.int64,
        )

@dataclass(frozen=True,slots=True)
class QuantityList:
    values: list[float]
    unit: str

    def to_QuantityData_list(
        self,
    )->list[QuantityData]:
        return [
            QuantityData(
                value=value,
                unit=self.unit
            )
            for value in self.values
        ]

@dataclass(frozen=True,slots=True)
class WideFormatDataFrames:
    dfs: list[pd.DataFrame]

    def to_long_format(
        self
    )->pd.DataFrame:
        attrs = self.dfs[0].attrs.copy()

        if any(df.attrs != attrs for df in self.dfs[1:]):
            raise ValueError(
                "結合するDataFrameのattrsが一致していません."
            )

        return pd.concat(
            self.dfs,
            ignore_index=True,
        )
