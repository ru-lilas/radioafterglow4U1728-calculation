import numpy as np
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
from typing import Self

from module import plot_utils
from module.mydataclasses import QuantityArray, YAMLReadable,QuantityData
from module import models
from module.plot_utils import ContourStyle
from module.types import FloatArray,FloatGrid
from module.inputs_as_dataclass import(
    PhysicalParameters
)
from module.parameter_table import LambdaPeakTable
from enum import StrEnum, auto
from matplotlib.axes import Axes

@dataclass(frozen=True,slots=True)
class ContourConfigures:
    a_wind: plot_utils.ContourConfigure
    beta_sh: plot_utils.ContourConfigure

@dataclass(frozen=True, slots=True)
class PlotConfig(YAMLReadable):
    layout: plot_utils.PlotLayoutConfig
    contours: ContourConfigures

class GridColumns(StrEnum):
    A_WIND = auto()
    BETA_SH = auto()
    PHI_PEAK = auto()
    FNU_PEAK = auto()

@dataclass(frozen=True,slots=True)
class ChevalierGrid:
    a_wind_grid: QuantityArray
    beta_sh_grid: FloatGrid
    eps_b: float
    eps_th: float
    mu: float
    mu_e: float
    distance: QuantityData

    @classmethod
    def from_yaml(
        cls,
        path:Path
    )->Self:
        inputs = PhysicalParameters.from_yaml(path)
        a_wind_grid,beta_sh_grid = inputs.build_meshgrid()
        return cls(
            a_wind_grid = a_wind_grid,
            beta_sh_grid = beta_sh_grid,
            eps_b = inputs.eps_b,
            eps_th = inputs.eps_th,
            mu = inputs.mu,
            mu_e = inputs.mu_e,
            distance = inputs.distance
        )

    @property
    def theta(self):
        return models.theta(
            eps_th=self.eps_th,
            mu=self.mu,
            mu_e=self.mu_e,
            beta_sh=self.beta_sh_grid
        )

    @property
    def phi_theta(self)->QuantityArray:
        phi_theta = models.phi_theta(
            theta = self.theta,
            eps_b = self.eps_b,
            a_wind = self.a_wind_grid
        )
        return QuantityArray(
            values = phi_theta.value,
            unit = phi_theta._unitstr
        )
    
    @property
    def tau_theta(self):
        return models.tau_theta(
            theta = self.theta,
            eps_b = self.eps_b,
            a_wind = self.a_wind_grid,
            beta_sh = self.beta_sh_grid,
            mu = self.mu,
            mu_e = self.mu_e
        )

    @property
    def lnu_theta(self):
        lnu = models.lnu_theta(
            theta = self.theta,
            eps_b = self.eps_b,
            a_wind = self.a_wind_grid,
            beta_sh = self.beta_sh_grid,
        )
        return QuantityArray(
            values = lnu.value,
            unit = lnu._unitstr
        )

    @property
    def doppler_delta(self):
        return models.doppler_delta(
            beta_sh = self.beta_sh_grid
        )

    def lambda_theta_peak(self,peak_table:LambdaPeakTable):
        return peak_table.interpolated_lambda_peak(self.tau_theta)

    def xm_peak(self,peak_table:LambdaPeakTable)->FloatArray:
        return peak_table.interpolated_xm_peak(self.tau_theta)

    def phi_peak(
        self,
        peak_table:LambdaPeakTable
    )->QuantityArray:
        phi_theta_arr:FloatArray = self.phi_theta.values
        unit = self.phi_theta.unit
        return QuantityArray(
            values = phi_theta_arr*self.xm_peak(peak_table),
            unit = unit
        )

    def lnu_peak(
        self,
        peak_table:LambdaPeakTable
    )->QuantityArray:
        lnu_theta_arr:FloatArray = self.lnu_theta.values
        unit = self.lnu_theta.unit
        return QuantityArray(
            values = lnu_theta_arr*self.lambda_theta_peak(peak_table),
            unit = unit
        )


@dataclass(frozen=True,slots=True)
class ChevalierGridBase:
    a_wind_arr: FloatArray
    beta_sh_arr: FloatArray
    phi_peak_arr: FloatArray
    fnu_peak_arr: FloatArray

    @property
    def df(self)->pd.DataFrame:
        df = pd.DataFrame({
            GridColumns.A_WIND: self.a_wind_arr,
            GridColumns.BETA_SH: self.beta_sh_arr,
            GridColumns.PHI_PEAK: self.phi_peak_arr,
            GridColumns.FNU_PEAK: self.fnu_peak_arr
        })
        if df.duplicated(
            subset=[GridColumns.A_WIND,GridColumns.BETA_SH]
        ).any():
            raise ValueError(
                "同じA_windとbeta_shの組が複数存在します."
            )
        else:
            return df

    @property
    def a_wind_axis(self)->FloatArray:
        return np.unique(self.a_wind_arr)

    @property
    def beta_sh_axis(self)->FloatArray:
        return np.unique(self.beta_sh_arr)

    @property
    def meshgrid(
        self,
    ) -> tuple[FloatArray, FloatArray]:

        a_wind_grid, beta_sh_grid = np.meshgrid(
            self.a_wind_axis,
            self.beta_sh_axis,
            indexing="xy",
        )

        return (
            a_wind_grid,
            beta_sh_grid,
        )

    def pivot_table(
        self,
        value: str,
    ) -> pd.DataFrame:
        return (
            self.df.pivot(
                index=GridColumns.BETA_SH,
                columns=GridColumns.A_WIND,
                values=value,
            )
            .reindex(
                index=self.beta_sh_axis,
                columns=self.a_wind_axis,
            )
        )

    def diagnose_pivot(
        self,
        value: str,
    ) -> None:
        df = self.df
        table = self.pivot_table(value)

        n_a_wind = self.a_wind_axis.size
        n_beta_sh = self.beta_sh_axis.size
        n_expected = n_a_wind * n_beta_sh
        n_actual = len(df)

        print(f"A_windの数: {n_a_wind}")
        print(f"beta_shの数: {n_beta_sh}")
        print(f"期待される組合せ数: {n_expected}")
        print(f"実際の行数: {n_actual}")
        print(f"pivot後のshape: {table.shape}")
        print(
            "元データ内のNaN数:",
            df[value].isna().sum(),
        )
        print(
            "pivot後のNaN数:",
            table.isna().sum().sum(),
        )

    def df_pivot(self,value:str):
        table = self.pivot_table(value)
        self.diagnose_pivot(value)

        if table.isna().any().any():
            raise ValueError(
                f"{value}のパラメータグリッドに欠損があります."
            )

        return table.to_numpy(dtype=np.float64)

    @property
    def phi_peak_grid(self):
        return self.df_pivot(GridColumns.PHI_PEAK)

    @property
    def fnu_peak_grid(self):
        return self.df_pivot(GridColumns.FNU_PEAK)

    def plot(
        self,
        ax:Axes,
        contourconfs: ContourConfigures
    ):
        a_wind_grid = self.meshgrid[0]
        beta_sh_grid = self.meshgrid[1]
        contour_awind = ax.contour(
            self.phi_peak_grid,
            self.fnu_peak_grid,
            a_wind_grid,
            levels=contourconfs.a_wind.levels.to_nparray(a_wind_grid),
            **contourconfs.a_wind.style.to_kwargs(),
        )
        contour_beta = ax.contour(
            self.phi_peak_grid,
            self.fnu_peak_grid,
            beta_sh_grid,
            levels=contourconfs.beta_sh.levels.to_nparray(beta_sh_grid),
            **contourconfs.beta_sh.style.to_kwargs(),
        )
        return

@dataclass(frozen=True,slots=True)
class ChevalierInputs:
    physical_parameters: PhysicalParameters
    peak_table: LambdaPeakTable

    @classmethod
    def import_from(
        cls,
        path_input: Path,
        path_peak_table: Path
    )->Self:
        return cls(
            physical_parameters = PhysicalParameters.from_yaml(path_input),
            peak_table = LambdaPeakTable.from_csv(path_peak_table)
        )

    @property
    def tau_theta(self)->FloatArray:
        return self.physical_parameters.tau_theta

    @property
    def xm_peak(self)->FloatArray:
        return self.peak_table.interpolated_xm_peak(self.tau_theta)

    @property
    def lambda_theta_peak(self)->FloatArray:
        return self.peak_table.interpolated_lambda_peak(self.tau_theta)

    @property
    def phi_peak(
        self,
    )->QuantityArray:
        phi_theta_arr:FloatArray = self.physical_parameters.phi_theta.value
        unit = self.physical_parameters.phi_theta._unitstr
        return QuantityArray(
            values = phi_theta_arr*self.xm_peak,
            unit = unit
        )

    @property
    def lnu_peak(
        self,
    )->QuantityArray:
        lnu_theta_arr:FloatArray = self.physical_parameters.lnu_theta.value
        unit = self.physical_parameters.lnu_theta._unitstr
        return QuantityArray(
            values = lnu_theta_arr*self.lambda_theta_peak,
            unit = unit
        )

    @property
    def fnu_peak_observerframe(
        self
    )->QuantityArray:
        distance = self.physical_parameters.distance
        doppler_delta = self.physical_parameters.doppler_delta
        fnu_sourceframe = models.lnu_into_fnu(
            lnu = self.lnu_peak,
            distance = distance
        )
        return models.fnu_sourceframe_into_fnu_observerframe(
            fnu_sourceframe=fnu_sourceframe,
            doppler_delta=doppler_delta
        )

    def build_chevalier_grid(
        self,
        a_wind_unit: str,
        phi_unit: str,
        fnu_unit: str
    )->ChevalierGridBase:
        return ChevalierGridBase(
            a_wind_arr=self.physical_parameters.a_wind.FloatArray_in(a_wind_unit),
            beta_sh_arr = self.physical_parameters.beta_sh,
            phi_peak_arr= self.phi_peak.FloatArray_in(phi_unit),
            fnu_peak_arr= self.fnu_peak_observerframe.FloatArray_in(fnu_unit)
        )
