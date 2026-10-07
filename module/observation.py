from pathlib import Path
from dataclasses import dataclass,replace
from typing import Self, cast, Any
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib import patches as pltpatch
from numpy.typing import NDArray
from module import dataframe_processors, dataframe_utils, mystatistics
from module.models import ColumnNames, MetadataNames
from module.mydataclasses import GridRange, QuantityArray1D, QuantityData,QuantityArray
from module.types import BoolGrid, FloatArray, FloatGrid
import numpy as np
import pandas as pd
from enum import StrEnum, auto
from module.inputs_as_dataclass import SamplingTimewindow
from module import plot_utils
from module.utils import FileReader

class Columns(StrEnum):
    T = auto()
    NU = auto()
    FNU = auto()
    FNU_ERR = auto()
    FNU_NET = auto()
    FNU_NET_ERR = auto()
    BG = auto()
    BG_ERR = auto()

@dataclass(frozen=True,slots=True)
class ChevalierComponent:
    t_peak: QuantityData
    t_peak_err: QuantityData
    fnu_peak: QuantityData
    fnu_peak_err: QuantityData
    nu: QuantityData

    @property
    def phi_peak(self)->QuantityData:
        phi_peak = self.t_peak.quantity*self.nu.quantity
        return QuantityData.from_Quantity(phi_peak)

    @property
    def phi_peak_err(self)->QuantityData:
        phi_peak_err = self.t_peak_err.quantity * self.nu.quantity
        return QuantityData.from_Quantity(phi_peak_err)

    def contains(
        self,
        phi_grid: FloatGrid,
        fnu_grid: FloatGrid,
        *,
        phi_unit: str,
        fnu_unit: str
    )->BoolGrid:
        phi_center = self.phi_peak.value_in(phi_unit)
        phi_error = self.phi_peak_err.value_in(phi_unit)

        fnu_center = self.fnu_peak.value_in(fnu_unit)
        fnu_error = self.fnu_peak_err.value_in(fnu_unit)

        phi_min = phi_center - phi_error
        phi_max = phi_center + phi_error
        fnu_min = fnu_center - fnu_error
        fnu_max = fnu_center + fnu_error

        return (
            (phi_grid >= phi_min)
            & (phi_grid <= phi_max)
            & (fnu_grid >= fnu_min)
            & (fnu_grid <= fnu_max)
        )

    def allowing_range(
        self,
        x_unit:str,
        y_unit:str
    )->GridRange:
        x_center = self.phi_peak.value_in(x_unit)
        x_err = self.phi_peak_err.value_in(x_unit)
        y_center = self.fnu_peak.value_in(y_unit)
        y_err = self.fnu_peak_err.value_in(y_unit)
        return GridRange(
            xmin = x_center - x_err,
            xmax = x_center + x_err,
            ymin = y_center - y_err,
            ymax = y_center + y_err
        )

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
        metadata = FileReader.keyvalue(path)
        return cls(**metadata)

    @property
    def t_bin_width(self)->QuantityData:
        return QuantityData(value=self.t_bin,unit=self.t_unit)


@dataclass(frozen=True, slots=True)
class Binning:
    bin_width: float
    t_center: FloatArray
    t_left: FloatArray
    t_right: FloatArray
    t_unit: str

    def to_df(
        self
    )->pd.DataFrame:
        df = pd.DataFrame({
            ColumnNames.T_LEFT: self.t_left,
            ColumnNames.T_CENTER: self.t_center,
            ColumnNames.T_RIGHT: self.t_right,
        })
        df.attrs[MetadataNames.T_UNIT] = self.t_unit
        df.attrs[MetadataNames.BIN_WIDTH] = self.bin_width
        return df

    @classmethod
    def from_df(
        cls,
        df:pd.DataFrame
    )->Self:
        attrs = df.attrs
        return cls(
            bin_width=attrs[MetadataNames.BIN_WIDTH],
            t_unit = attrs[MetadataNames.T_UNIT],
            t_center = df[ColumnNames.T_CENTER].to_numpy(),
            t_right = df[ColumnNames.T_RIGHT].to_numpy(),
            t_left = df[ColumnNames.T_LEFT].to_numpy(),
        )

@dataclass(frozen=True, slots=True)
class ObservedLightcurve:
    metadata: LightcurveMetadata
    nu_value: float
    df: pd.DataFrame

    @property
    def nu(self)->QuantityData:
        return QuantityData(
            value=self.nu_value,
            unit=self.metadata.nu_unit
        )

    @property
    def fnu_background(self):
        return QuantityArray1D(
            values = dataframe_processors.convert_ndarray(
                self.df,
                column=Columns.BG
            ),
            unit=self.metadata.fnu_unit
        )
    @property
    def fnu_background_err(self):
        return QuantityArray1D(
            values = dataframe_processors.convert_ndarray(
                self.df,
                column=Columns.BG_ERR
            ),
            unit=self.metadata.fnu_unit
        )

    @property
    def fnu_net(self)->QuantityArray:
        return QuantityArray(
            values = dataframe_utils.extract_column_as_ndarray(
                self.df,Columns.FNU_NET
            ),
            unit = self.metadata.fnu_unit
        )

    @property
    def fnu_net_err(self)->QuantityArray:
        return QuantityArray(
            values = dataframe_utils.extract_column_as_ndarray(
                self.df,Columns.FNU_NET_ERR
            ),
            unit = self.metadata.fnu_unit
        )

    @property
    def fnu_total(self):
        values = dataframe_processors.convert_ndarray(
            df = self.df,
            column= Columns.FNU,
        )
        return QuantityArray1D(
            values = values,
            unit = self.metadata.fnu_unit
        )

    @property
    def fnu_total_err(self):
        values = dataframe_processors.convert_ndarray(
            df = self.df,
            column= Columns.FNU_ERR,
        )
        return QuantityArray1D(
            values = values,
            unit = self.metadata.fnu_unit
        )

    def time_bin_bounds(
        self,
        timewindow: SamplingTimewindow,
    ) -> Binning:
        t: FloatArray = np.asarray(
            self.df[Columns.T],
            dtype=np.float64,
        )
        t_unit = self.metadata.t_unit

        if t.size < 2:
            raise ValueError(
                "bin幅の推定には2点以上の時刻が必要です."
            )

        dt: FloatArray = np.diff(t)

        if np.any(dt <= 0.0):
            raise ValueError(
                "時刻は単調増加でなければなりません."
            )

        if not np.allclose(dt, dt[0]):
            raise ValueError(
                "観測時刻が等間隔ではありません."
            )

        bin_width = float(dt[0])
        t_min = float(
            timewindow.t_min.value_in(t_unit)
        )
        t_max = float(
            timewindow.t_max.value_in(t_unit)
        )

        inside = (t >= t_min) & (t <= t_max)
        t_center = t[inside]

        if t_center.size == 0:
            raise ValueError(
                "指定された時間範囲に観測点がありません."
            )

        return Binning(
            bin_width=bin_width,
            t_center=t_center,
            t_left=t_center - 0.5 * bin_width,
            t_right=t_center + 0.5 * bin_width,
            t_unit=t_unit,
        )

    def to_FloatArray(self, column:str):
        return dataframe_processors.convert_ndarray(
            self.df,
            column
        )

    def select_timewindow(
        self,
        *,
        t_min:float,
        t_max:float
    ) -> Self:

        t_unit = self.metadata.t_unit

        #=== ここから下はSamplingTimewindowクラスに記述するべきでは ===#
        # t_min = t_min.value_in(t_unit)
        # t_max = t_max.value_in(t_unit)
        #
        # if t_max <= t_min:
        #     raise ValueError(
        #         "timewindow.maxはminより大きくしてください."
        #     )
        #=== ここまで ===#

        t = self.to_FloatArray(Columns.T)

        inside = (t >= t_min ) & ( t <= t_max )

        if not np.any(inside):
            raise ValueError(
            "指定された時間範囲に観測データがありません."
        )

        df_selected = cast(
            pd.DataFrame,
            self.df.loc[inside].reset_index(drop=True).copy()
        )

        return replace(
            self,
            df=df_selected
        )
    @property
    def t(self):
        return QuantityArray1D(
            values = self.to_FloatArray(Columns.T),
            unit = self.metadata.t_unit
        )

    @property
    def peak(self)->ChevalierComponent:
        metadata = self.metadata
        t_width = metadata.t_bin_width
        nu = self.nu
        fnu_unit = metadata.fnu_unit
        t_unit = metadata.t_unit

        df_top2 = self.df.nlargest(2,Columns.FNU_NET)
        fnu_net     = dataframe_utils.extract_column_as_ndarray(df_top2,Columns.FNU_NET)
        fnu_net_err = dataframe_utils.extract_column_as_ndarray(df_top2,Columns.FNU_NET_ERR)
        t           = dataframe_utils.extract_column_as_ndarray(df_top2,Columns.T)

        fnu_peak, fnu_peak_err = mystatistics.calculate_muhat(fnu_net,fnu_net_err)
        t_peak = QuantityData(float(np.mean(t)),unit=t_unit)
        t_peak_width = np.ptp(t) + t_width.value
        t_peak_err = QuantityData(t_peak_width / 2.0, unit=t_unit)
        return ChevalierComponent(
            t_peak = t_peak,
            t_peak_err = t_peak_err,
            fnu_peak = QuantityData(fnu_peak,fnu_unit),
            fnu_peak_err = QuantityData(fnu_peak_err,fnu_unit),
            nu=nu
        )
@dataclass(frozen=True, slots=True)
class LongformatLightcurve:
    metadata: LightcurveMetadata
    df_long: pd.DataFrame

    @classmethod
    def from_csv(
        cls,
        path: Path,
    ) -> Self:
        df_long = FileReader.table_from_csv(path)
        metadata = df_long.attrs

        if not all(isinstance(key, str) for key in metadata):
            raise TypeError("All dictionary keys must be strings.")
        else:
            metadata = cast(dict[str, Any], metadata)

        required_columns = {
            Columns.T,
            Columns.FNU,
            Columns.FNU_ERR,
            Columns.NU,
            Columns.FNU_NET,
            Columns.FNU_NET_ERR,
            Columns.BG,
            Columns.BG_ERR,
        }

        missing_columns = required_columns - set(df_long.columns)

        if missing_columns:
            raise ValueError(
                f"必要な列がありません: {sorted(missing_columns)}"
            )

        return cls(
            metadata=LightcurveMetadata(**metadata),
            df_long=df_long,
        )

    def group_by_frequency(
        self,
    ) -> dict[float, ObservedLightcurve]:
        lightcurves: dict[float, ObservedLightcurve] = {}

        for nu, df_nu in self.df_long.groupby(
            Columns.NU,
            sort=True,
        ):
            nu_value = cast(float,nu)

            lightcurves[nu_value] = ObservedLightcurve(
                metadata=self.metadata,
                nu_value=nu_value,
                df=df_nu.reset_index(drop=True),
            )

        return lightcurves

    def select_frequency(
        self,
        nu:QuantityData,
        *,
        unit:str
    )->ObservedLightcurve:
        nu_value = nu.value_in(unit)
        lcs = self.group_by_frequency()
        return lcs[nu_value]
