from matplotlib.contour import QuadContourSet
import numpy as np
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
from typing import Self

from module import dataframe_utils, plot_utils
from module.mydataclasses import QuantityArray, YAMLReadable,QuantityData
from module import models
from module.observation import ChevalierScatter
from module.plot_utils import ContourStyle, MetadataConfig
from module.types import FloatArray,FloatGrid
from module.inputs_as_dataclass import(
    ModelParameterGrid
)
from module.parameter_table import LambdaPeakTable
from enum import StrEnum, auto
from matplotlib.axes import Axes
from module.compute_lightcurve import AllowningParameters
from module.utils import FileReader

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
    metadata: dict[str,MetadataConfig]

class GridColumns(StrEnum):
    A_WIND = auto()
    BETA_SH = auto()
    PHI_PEAK = auto()
    FNU_PEAK = auto()

@dataclass(frozen=True,slots=True)
class ABGrid:
    a_wind_grid: FloatGrid
    beta_sh_grid: FloatGrid

@dataclass(frozen=True,slots=True)
class ChevalierGrid:
    a_wind_arr: QuantityArray
    beta_sh_arr: FloatGrid
    eps_b: float
    eps_th: float
    mu: float
    mu_e: float
    distance: QuantityData

    def __post_init__(self) -> None:
        a_wind_shape = self.a_wind_arr.values.shape
        beta_sh_shape = self.beta_sh_arr.shape

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
        inputs = ModelParameterGrid.from_yaml(path)
        a_wind_grid,beta_sh_grid = inputs.build_meshgrid()
        return cls(
            a_wind_arr = a_wind_grid,
            beta_sh_arr = beta_sh_grid,
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
            beta_sh=self.beta_sh_arr
        )

    @property
    def phi_theta(self)->QuantityArray:
        phi_theta = models.phi_theta(
            theta = self.theta,
            eps_b = self.eps_b,
            a_wind = self.a_wind_arr
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
            a_wind = self.a_wind_arr,
            beta_sh = self.beta_sh_arr,
            mu = self.mu,
            mu_e = self.mu_e
        )

    @property
    def lnu_theta(self):
        lnu = models.lnu_theta(
            theta = self.theta,
            eps_b = self.eps_b,
            a_wind = self.a_wind_arr,
            beta_sh = self.beta_sh_arr,
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
            beta_sh = self.beta_sh_arr
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
    )->ABGrid:
        a_wind_grid, beta_sh_grid = np.broadcast_arrays(
            self.a_wind_arr.values,
            self.beta_sh_arr,
        )
        return ABGrid(
            a_wind_grid = np.asarray(a_wind_grid,dtype=np.float64),
            beta_sh_grid = np.asarray(beta_sh_grid,dtype=np.float64),
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

        abgrid = self.parameter_grids
        a_wind_grid = abgrid.a_wind_grid
        beta_sh_grid = abgrid.beta_sh_grid

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
        ax.clabel(
            contour_a_wind,
            **contourconfs.a_wind.label.to_kwargs(),
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
        ax.clabel(
            contour_beta_sh,
            **contourconfs.beta_sh.label.to_kwargs(),
        )

        return (
                contour_a_wind,
                contour_beta_sh,
            )

    def plot_metadata(
        self,
        ax: Axes,
        plotconf: PlotConfig
    ):
        strings = []
        conf_metadata = plotconf.metadata
        conf_eps_b = conf_metadata["eps_b"]
        conf_eps_th = conf_metadata["eps_th"]
        conf_mu = conf_metadata["mu"]
        conf_mu_e = conf_metadata["mu_e"]
        strings.append(conf_eps_b.format_value(self.eps_b))
        strings.append(conf_eps_th.format_value(self.eps_th))
        strings.append(conf_mu.format_value(self.mu))
        strings.append(conf_mu_e.format_value(self.mu_e))
        ax.set_title(
            ", ".join(strings),
            fontsize=16
        )
    
    def select_parameters_in(
        self,
        observation: ChevalierScatter,
        peak_table: LambdaPeakTable,
        *,
        phi_unit: str,
        fnu_unit: str,
    )->AllowningParameters:
        phi_grid = (
            self.phi_peak(peak_table)
            .to_value_in(phi_unit)
        )
        fnu_grid = (
            self.fnu_peak_observerframe(peak_table)
            .to_value_in(fnu_unit)
        )
        abgrid = self.parameter_grids

        mask = observation.contains(
            phi_grid,
            fnu_grid,
            phi_unit=phi_unit,
            fnu_unit=fnu_unit,
        )

        if mask.shape != abgrid.a_wind_grid.shape:
            raise ValueError(
                "包含判定マスクとパラメータグリッドの"
                "shapeが一致しません. "
                f"mask={mask.shape}, "
                f"beta_sh={abgrid.a_wind_grid.shape}"
            )

        phi_theta_q = self.phi_theta
        phi_theta = QuantityArray(
            phi_theta_q.values[mask],
            phi_theta_q.unit
        )
        lnu_theta_q = self.lnu_theta
        lnu_theta = QuantityArray(
            lnu_theta_q.values[mask],
            lnu_theta_q.unit
        )
        tau_theta = self.tau_theta[mask]
        doppler_delta = models.doppler_delta(abgrid.beta_sh_grid)[mask]

        return AllowningParameters(
            a_wind = QuantityArray(
                values=abgrid.a_wind_grid[mask],
                unit=self.a_wind_arr.unit
            ),
            beta_sh = abgrid.beta_sh_grid[mask],
            phi_peak = QuantityArray(
                values=phi_grid[mask],
                unit=phi_unit
            ),
            fnu_peak = QuantityArray(
                values=fnu_grid[mask],
                unit=fnu_unit
            ),
            tau_theta = tau_theta,
            lnu_theta=lnu_theta,
            phi_theta=phi_theta,
            eps_b=self.eps_b,
            eps_th=self.eps_th,
            distance=self.distance,
            doppler_delta=doppler_delta,
            mu=self.mu,
            mu_e=self.mu_e
        )
