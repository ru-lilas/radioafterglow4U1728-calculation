from matplotlib.contour import QuadContourSet
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

@dataclass(frozen=True,slots=True)
class Styles:
    obs_center: plot_utils.ScatterStyle
    obs_range: plot_utils.RectangleStyle

@dataclass(frozen=True, slots=True)
class PlotConfig(YAMLReadable):
    layout: plot_utils.PlotLayoutConfig
    contours: ContourConfigures
    styles: Styles

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

    def __post_init__(self) -> None:
        a_wind_shape = self.a_wind_grid.values.shape
        beta_sh_shape = self.beta_sh_grid.shape

        if len(a_wind_shape) != 2:
            raise ValueError(
                "a_wind_gridは2次元でなければなりません. "
                f"shape={a_wind_shape}"
            )

        if len(beta_sh_shape) != 2:
            raise ValueError(
                "beta_sh_gridは2次元でなければなりません. "
                f"shape={beta_sh_shape}"
            )

        try:
            np.broadcast_shapes(
                a_wind_shape,
                beta_sh_shape,
            )
        except ValueError as error:
            raise ValueError(
                "a_wind_gridとbeta_sh_gridを"
                "ブロードキャストできません. "
                f"a_wind={a_wind_shape}, "
                f"beta_sh={beta_sh_shape}"
            ) from error

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
        if phi_theta.unit is None:
            raise ValueError("phi_thetaの単位がありません.")
        return QuantityArray(
            values = phi_theta.value,
            unit = phi_theta.unit.to_string()
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
        if lnu.unit is None:
            raise ValueError("lnuの単位がありません.")
        return QuantityArray(
            values = lnu.value,
            unit = lnu.unit.to_string()
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
        phi_theta_quantity = self.phi_theta
        return QuantityArray(
            values = phi_theta_quantity.values
                *self.xm_peak(peak_table),
            unit = phi_theta_quantity.unit
        )

    def lnu_peak(
        self,
        peak_table:LambdaPeakTable
    )->QuantityArray:
        lnu_theta_quantity = self.lnu_theta
        return QuantityArray(
            values = lnu_theta_quantity.values
                *self.lambda_theta_peak(peak_table),
            unit = lnu_theta_quantity.unit
        )

    def fnu_peak_observerframe(
        self,
        peak_table:LambdaPeakTable
    )->QuantityArray:
        fnu_sourceframe = models.lnu_into_fnu(
            lnu = self.lnu_peak(peak_table),
            distance = self.distance
        )
        return models.fnu_sourceframe_into_fnu_observerframe(
            fnu_sourceframe=fnu_sourceframe,
            doppler_delta=self.doppler_delta
        )

    @property
    def parameter_grids(
        self
    )->tuple[FloatArray,FloatArray]:
        a_wind_grid, beta_sh_grid = np.broadcast_arrays(
            self.a_wind_grid.values,
            self.beta_sh_grid,
        )
        return (
            np.asarray(
                a_wind_grid,
                dtype=np.float64
            ),
            np.asarray(
                beta_sh_grid,
                dtype=np.float64
            ),
        )

    def plot(
        self,
        ax:Axes,
        peak_table: LambdaPeakTable,
        contourconfs: ContourConfigures,
        *,
        phi_unit: str,
        fnu_unit: str,
    )->tuple[QuadContourSet,QuadContourSet]:
        phi_peak = (
            self.phi_peak(peak_table)
            .to_value_in(phi_unit)
        )
        fnu_peak = (
            self.fnu_peak_observerframe(peak_table)
            .to_value_in(fnu_unit)
        )

        a_wind_grid,beta_sh_grid = self.parameter_grids

        if not (
            phi_peak.shape
            == fnu_peak.shape
            == a_wind_grid.shape
            == beta_sh_grid.shape
        ):
            raise ValueError(
                "等高線に使用するグリッドのshapeが"
                "一致していません. "
                f"phi_peak={phi_peak.shape}, "
                f"fnu_peak={fnu_peak.shape}, "
                f"a_wind={a_wind_grid.shape}, "
                f"beta_sh={beta_sh_grid.shape}"
            )

        contour_a_wind = ax.contour(
            phi_peak,
            fnu_peak,
            a_wind_grid,
            levels=(
                contourconfs.a_wind.levels
                .to_nparray(a_wind_grid)
            ),
            **contourconfs.a_wind.style.to_kwargs(),
        )

        contour_beta_sh = ax.contour(
            phi_peak,
            fnu_peak,
            beta_sh_grid,
            levels=(
                contourconfs.beta_sh.levels
                .to_nparray(beta_sh_grid)
            ),
            **contourconfs.beta_sh.style.to_kwargs(),
        )

        return (
                contour_a_wind,
                contour_beta_sh,
            )
