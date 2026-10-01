from collections.abc import Mapping
from pathlib import Path
from dataclasses import asdict, dataclass, fields, replace
from typing import Self, cast, Any, Hashable
from numpy.typing import NDArray
from matplotlib.figure import Figure
from matplotlib.patches import StepPatch
from module import plot_utils, quantity_converter
from module import models,dataframe_utils
from module import mystatistics
from module.models import ChiSquaredParameters, ColumnNames, DefaultUnits, FluxName, ScenarioParameter
from module.mydataclasses import IntVector, QuantityArray, QuantityArray1D, QuantityData, QuantityGrid,FloatGrid, YAMLReadable
from module.mystatistics import CalculateChiSquared, MinimumChiSquared
from module.plot_utils import LineStyle, MetadataConfig
from module.types import FloatArray, FloatArray1D
import astropy.units as u
import numpy as np
import pandas as pd
from module.utils import DataFrameUtils, FileWriter, Integrator,FileReader
from module import dataframe_processors, observation
from enum import StrEnum, auto
from matplotlib.axes import Axes
from cattrs import Converter
from module import inputs_as_dataclass
from module.models import (
    MetadataNames
)

DATACLASS_CONVERTER = Converter()

class LcColumns(StrEnum):
    T = auto()
    NU = auto()
    FNU_OBSERVER_FRAME = auto()
    FNU_ERR = auto()
    FNU_NET = auto()
    FNU_NET_ERR = auto()
    T_UNIT = auto()
    FNU_UNIT = auto()
    T_CENTER = auto()
    T_LEFT = auto()
    T_RIGHT = auto()
    FNU_AVERAGED = auto()
    FNU_TRANSIENT = auto()
    FNU_PERSISTENT = auto()
    FNU_PERSISTENT_ERR = auto()
    NU_VALUE = auto()
    NU_UNIT = auto()

class ParameterNames(StrEnum):
    A_WIND = auto()
    BETA_SH = auto()
    PHI_PEAK = auto()
    FNU_PEAK = auto()
    PHI_THETA = auto()
    LNU_THETA = auto()
    TAU_THETA = auto()
    EPS_B = auto()
    EPS_TH = auto()
    MU = auto()
    MU_E = auto()
    DISTANCE_VALUE = auto()
    DISTANCE_UNIT = auto()
    PHI_UNIT = auto()
    LNU_UNIT = auto()
    FNU_UNIT = auto()
    A_WIND_UNIT = auto()
    DOPPLER_DELTA = auto()
    CHI2 = auto()

@dataclass(frozen=True,slots=True)
class ChiSquaredResult:
    model_index: int
    nu: float
    chi_squared: float
    ndof: int

    def to_series(
        self
    )->pd.Series:
        return pd.Series(asdict(self))

    @classmethod
    def from_series(
        cls,
        row:pd.Series
    )->Self:
        return cls(**row.to_dict())

    @classmethod
    def calculate(
        cls,
        model_index: int,
        nu_value: float,
        fnu_model: QuantityArray1D,
        fnu_data: QuantityArray1D,
        fnu_err: QuantityArray1D,
        *,
        n_params: int,
        fnu_unit: str = DefaultUnits.fnu,
    )->Self:
        n_data = fnu_model.length

        if not (
            fnu_data.length
            == fnu_err.length
            == n_data
        ):
            raise ValueError(
            "モデル, 観測値, 観測誤差の"
            "配列長が一致していません."
        )
        chi2 = CalculateChiSquared.from_vector(
            model = fnu_model.to_floatarray1d(fnu_unit),
            observed = fnu_data.to_floatarray1d(fnu_unit),
            error = fnu_err.to_floatarray1d(fnu_unit)
        )

        return cls(
            model_index=model_index,
            nu=nu_value,
            chi_squared=chi2,
            ndof=n_data-n_params
        )
@dataclass(frozen=True,slots=True)
class BinnedLightcurveWithBackground:
    model_index: int
    nu: QuantityData
    t_center: QuantityArray1D
    t_left: QuantityArray1D
    t_right: QuantityArray1D
    fnu_with_bg: QuantityArray1D

    @property
    def bin_edges(self):
        t_unit = self.t_left.unit
        t_left = self.t_left.to_value_in(t_unit)
        t_right = self.t_right.to_value_in(t_unit)
        bin_edges = np.concatenate(
            (
                t_left,
                np.array([t_right[-1]]),
            )
        )
        return QuantityArray1D(
            values=bin_edges,
            unit=t_unit
        )

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame
    )-> Self:
        attrs = df.attrs
        nu = QuantityData(
            value=attrs[ColumnNames.NU],
            unit=attrs[MetadataNames.NU_UNIT],
        )
        return cls(
            model_index = attrs[ColumnNames.MODEL_INDEX],
            nu = nu,
            t_center = QuantityArray1D.from_df(df,ColumnNames.T_CENTER,MetadataNames.T_UNIT),
            t_right = QuantityArray1D.from_df(df,ColumnNames.T_RIGHT,MetadataNames.T_UNIT),
            t_left = QuantityArray1D.from_df(df,ColumnNames.T_LEFT,MetadataNames.T_UNIT),
            fnu_with_bg = QuantityArray1D.from_df(df,FluxName.FNU_WITH_BG,MetadataNames.FNU_UNIT),
        )

@dataclass(frozen=True,slots=True)
class BinnedLightcurve:
    model_index: int
    nu: QuantityData
    t_center: QuantityArray1D
    t_left: QuantityArray1D
    t_right: QuantityArray1D
    fnu_no_bg: QuantityArray1D

    def fnu_with_bg(
        self,
        fnu_bg: QuantityArray1D
    )->QuantityArray1D:
        return QuantityArray1D.from_quantity(
            self.fnu_no_bg.quantity
            + fnu_bg.quantity
        )

    @property
    def bin_edges(self):
        t_unit = self.t_left.unit
        t_left = self.t_left.to_value_in(t_unit)
        t_right = self.t_right.to_value_in(t_unit)
        bin_edges = np.concatenate(
            (
                t_left,
                np.array([t_right[-1]]),
            )
        )
        return QuantityArray1D(
            values=bin_edges,
            unit=t_unit
        )

    def to_df(
        self,
        fnu_bg: QuantityArray1D,
        *,
        nu_unit: str = DefaultUnits.nu,
        t_unit: str = DefaultUnits.t,
        fnu_unit: str = DefaultUnits.fnu,
    )->pd.DataFrame:
        df = pd.DataFrame({
            models.ColumnNames.MODEL_INDEX: self.model_index,
            models.ColumnNames.NU: self.nu.value_in(nu_unit),
            models.ColumnNames.T_CENTER: self.t_center.to_value_in(t_unit),
            models.ColumnNames.T_RIGHT: self.t_right.to_value_in(t_unit),
            models.ColumnNames.T_LEFT: self.t_left.to_value_in(t_unit),
            models.FluxName.FNU_WITH_BG: self.fnu_with_bg(fnu_bg).to_value_in(fnu_unit),
        })
        df.attrs[MetadataNames.NU_UNIT] = nu_unit
        df.attrs[MetadataNames.FNU_UNIT] = fnu_unit
        df.attrs[MetadataNames.T_UNIT] = t_unit
        return df

@dataclass(frozen=True,slots=True)
class ContinuousLightcurve:
    model_index: int
    nu: QuantityData
    t: QuantityArray1D
    fnu_net: QuantityArray1D

    def insert_zero_boundary(
        self
    )->Self:
        """
            配列の左端に時刻ゼロでの境界条件を挿入する
        """
        t_values = self.t.values
        fnu_values = self.fnu_net.values

        if t_values[0] == 0.0:
            return self

        if t_values[0] < 0.0:
            raise ValueError(
                "計算光度曲線に負の時刻が含まれています."
                f"t_min={t_values[0]}"
            )

        return type(self)(
            model_index=self.model_index,
            t=QuantityArray1D(
                values=np.concatenate((
                    np.array([0.0], dtype=np.float64),
                    t_values,
                )),
                unit=self.t.unit,
            ),
            fnu_net=QuantityArray1D(
                values=np.concatenate((
                    np.array([0.0], dtype=np.float64),
                    fnu_values,
                )),
                unit=self.fnu_net.unit,
            ),
            nu=self.nu,
        )

    @staticmethod
    def validate_time_coverage(
        t_model: QuantityArray1D,
        binning: observation.Binning,
    ) -> None:
        t_model_arr = t_model.values

        if t_model_arr.size < 2:
            raise ValueError(
                "理論光度曲線には2点以上の時刻が必要です."
            )

        if (
            binning.t_left[0] < t_model_arr[0]
            and not np.isclose(
                binning.t_left[0],
                t_model_arr[0],
            )
        ):
            raise ValueError(
                "理論光度曲線が最初の観測binを覆っていません."
            )

        if (
            binning.t_right[-1] > t_model_arr[-1]
            and not np.isclose(
                binning.t_right[-1],
                t_model_arr[-1],
            )
        ):
            raise ValueError(
                "理論光度曲線が最後の観測binを覆っていません."
            )

    def bin_average(
        self,
        bin_config: observation.Binning,
        drop_incomplete_bin: bool = True
    )->BinnedLightcurve:

        t_unit = bin_config.t_unit
        fnu_unit = self.fnu_net.unit

        ## 計算光度曲線が観測の時間範囲を覆っているかcheck
        t = self.t
        fnu = self.fnu_net
        self.validate_time_coverage(t,bin_config)

        width = bin_config.bin_width
        t_model = t.to_value_in(t_unit)
        fnu_model = fnu.to_value_in(fnu_unit)

        if np.any(np.diff(t_model) <= 0.0):
            raise ValueError(
                "計算データの時刻配列が単調増加でありません."
            )


        t_center = bin_config.t_center
        t_left = bin_config.t_left
        t_right = bin_config.t_right
        num_bins = t_center.size

        if not (
            t_left.size
            == t_right.size
            == num_bins
        ):
            raise ValueError(
                "観測binの配列長が一致しません."
            )

         # 計算時間範囲に完全に含まれるbin
        covered = (
            (t_left >= t_model[0])
            & (t_right <= t_model[-1])
        )

        if drop_incomplete_bin:
            t_center = t_center[covered]
            t_left = t_left[covered]
            t_right = t_right[covered]

            if t_center.size == 0:
                raise ValueError(
                    "計算時間範囲に完全に含まれるbinがありません."
                )

        def interpolate_flux(
            t_evaluate: FloatArray,
        ) -> FloatArray:
            return np.interp(
                t_evaluate,
                t_model,
                fnu_model,
            )

        # shapeはいずれも(M, B)
        fnu_left = interpolate_flux(t_left)
        fnu_right = interpolate_flux(t_right)

        # shapeはいずれも(M, B)
        fnu_left = interpolate_flux(t_left)
        fnu_right = interpolate_flux(t_right)

        fnu_average_by_bin: list[float] = []

        for i, (left, right) in enumerate(
            zip(t_left, t_right, strict=True)
        ):
            inside = (
                (t_model > left)
                & (t_model < right)
            )

            t_bin = np.concatenate((
                np.array([left], dtype=np.float64),
                t_model[inside],
                np.array([right], dtype=np.float64),
            ))

            fnu_bin = np.concatenate((
                np.array([fnu_left[i]], dtype=np.float64),
                fnu_model[inside],
                np.array([fnu_right[i]], dtype=np.float64),
            ))

            flux_integral = np.trapezoid(
                fnu_bin,
                x=t_bin,
            )

            fnu_average_by_bin.append(
                float(flux_integral / (right - left))
            )

        fnu_average = QuantityArray1D(
            values=np.asarray(
                fnu_average_by_bin,
                dtype=np.float64,
            ),
            unit=fnu_unit,
        )

        return BinnedLightcurve(
            model_index= self.model_index,
            nu = self.nu,
            t_center=QuantityArray1D(t_center,t_unit),
            t_left=QuantityArray1D(t_left,t_unit),
            t_right=QuantityArray1D(t_right,t_unit),
            fnu_no_bg=fnu_average,
        )

    @classmethod
    def from_df(
        cls,
        model_index: int,
        nu: QuantityData,
        df: pd.DataFrame
    )-> Self:
        fnu_unit = df.attrs[MetadataNames.FNU_UNIT]
        t_unit = df.attrs[MetadataNames.T_UNIT]
        return cls(
            model_index=model_index,
            nu=nu,
            fnu_net=QuantityArray1D(
                df[models.ColumnNames.FNU_TOTAL]
                    .to_numpy(dtype=np.float64),
                unit=fnu_unit,
            ),
            t=QuantityArray1D(
                df[models.ColumnNames.T]
                    .to_numpy(dtype=np.float64),
                unit=t_unit,
            ),
        )

    def fnu_with_bg(
        self,
        fnu_bg:QuantityData
    )->QuantityArray1D:
        return QuantityArray1D.from_quantity(
            self.fnu_net.quantity + fnu_bg.quantity
        )
