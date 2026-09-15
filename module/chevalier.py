from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure
import numpy as np
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
from typing import Any, Hashable, Self

from module import dataframe_utils, plot_utils
from module.mydataclasses import (
    QuantityArray, YAMLReadable,QuantityData,QuantityArray1D,
    QuantityGrid
)
from module import models
from module.observation import ChevalierScatter
from module.plot_utils import ColormapConfigure, ContourStyle, MetadataConfig
from module.types import FloatArray,FloatGrid
from module.parameter_table import ModelParameterNames,ModelColumnNames
from enum import StrEnum, auto
from matplotlib.axes import Axes
from module.compute_lightcurve import AllowningParameters
from module.utils import FileReader

@dataclass(frozen=True,slots=True)
class ContourConfigures:
    a_wind: plot_utils.ContourConfigure
    beta_sh: plot_utils.ContourConfigure
    phi_th: plot_utils.ContourConfigure
    theta0: plot_utils.ContourConfigure

@dataclass(frozen=True,slots=True)
class ColormapConfigures:
    theta: plot_utils.ColormapConfigure
    tau_th: plot_utils.ColormapConfigure
    phi_th: plot_utils.ColormapConfigure
    lnu_th: plot_utils.ColormapConfigure

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

@dataclass(frozen=True, slots=True)
class DebagPlotConfig(YAMLReadable):
    layout: plot_utils.MultiPlotLayoutConfig
    colormaps: ColormapConfigures
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
class ChevalierMetadata:
    eps_b: float
    eps_th: float
    mue_mu: float
    distance: QuantityData
    phi_unit: str
    lnu_unit: str
    fnu_unit: str
    a_wind_unit: str

    @classmethod
    def from_attrs(
        cls,
        attrs:dict[Hashable,Any]
    )->Self:
        return cls(
            eps_b = attrs[ModelParameterNames.EPS_B],
            eps_th = attrs[ModelParameterNames.EPS_TH],
            mue_mu = attrs[ModelParameterNames.MU_E_MU],
            distance = QuantityData(
                value=attrs[ModelParameterNames.DISTANCE_VALUE],
                unit =attrs[ModelParameterNames.DISTANCE_UNIT]
            ),
            a_wind_unit = attrs[ModelParameterNames.A_WIND_UNIT],
            phi_unit = attrs[ModelParameterNames.PHI_UNIT],
            lnu_unit = attrs[ModelParameterNames.LNU_UNIT],
            fnu_unit = attrs[ModelParameterNames.FNU_UNIT],
        )

@dataclass(frozen=True,slots=True)
class ChevalierGrid:
    metadata: ChevalierMetadata
    phi_peak: QuantityGrid
    fnu_peak: QuantityGrid
    phi_th: QuantityGrid
    theta0: QuantityGrid
    theta: QuantityGrid
    tau_th: QuantityGrid
    lnu_th: QuantityGrid
    
    # def __post_init__(self) -> None:
    #     a_wind_shape = self.a_wind_arr.values.shape
    #     beta_sh_shape = self.beta_sh_arr.shape
    #
    #     if len(a_wind_shape) != 2:
    #         raise ValueError(
    #             "a_wind_gridは2次元でなければなりません. "
    #             f"shape={a_wind_shape}"
    #         )
    #
    #     if len(beta_sh_shape) != 2:
    #         raise ValueError(
    #             "beta_sh_gridは2次元でなければなりません. "
    #             f"shape={beta_sh_shape}"
    #         )
    #
    #     try:
    #         np.broadcast_shapes(
    #             a_wind_shape,
    #             beta_sh_shape,
    #         )
    #     except ValueError as error:
    #         raise ValueError(
    #             "a_wind_gridとbeta_sh_gridを"
    #             "ブロードキャストできません. "
    #             f"a_wind={a_wind_shape}, "
    #             f"beta_sh={beta_sh_shape}"
    #         ) from error

    @classmethod
    def from_csv(
        cls,
        path:Path
    ):
        df = FileReader.table_from_csv(path,idx="index")
        metadata = ChevalierMetadata.from_attrs(df.attrs)
        phi_unit = metadata.phi_unit
        fnu_unit = metadata.fnu_unit
        lnu_unit = metadata.lnu_unit

        return cls(
            metadata = metadata,
            phi_peak = QuantityGrid.from_df(
                df=df,
                unit=phi_unit,
                column=ModelColumnNames.PHI_PEAK
            ),         
            fnu_peak = QuantityGrid.from_df(
                df=df,
                unit=fnu_unit,
                column=ModelColumnNames.FNU_PEAK
            ),
            phi_th = QuantityGrid.from_df(
                df=df,
                unit=phi_unit,
                column=ModelParameterNames.PHI_TH
            ),
            theta0 = QuantityGrid.from_df(
                df=df,
                unit="",
                column=ModelParameterNames.THETA0
            ),
            theta = QuantityGrid.from_df(
                df=df,
                unit="",
                column=ModelColumnNames.THETA
            ),
            tau_th = QuantityGrid.from_df(
                df=df,
                unit="",
                column=ModelColumnNames.TAU_THETA
            ),
            lnu_th = QuantityGrid.from_df(
                df=df,
                unit=lnu_unit,
                column=ModelColumnNames.LNU_THETA
            )
        )

    # @property
    # def parameter_grids(
    #     self
    # )->ABGrid:
    #     a_wind_grid, beta_sh_grid = np.broadcast_arrays(
    #         self.a_wind_arr.values,
    #         self.beta_sh_arr,
    #     )
    #     return ABGrid(
    #         a_wind_grid = np.asarray(a_wind_grid,dtype=np.float64),
    #         beta_sh_grid = np.asarray(beta_sh_grid,dtype=np.float64),
    #     )
    #
    # def plot_main_contour(
    #     self,
    #     ax:Axes,
    #     contourconfs: ContourConfigures,
    #     *,
    #     phi_unit: str,
    #     fnu_unit: str,
    # )->tuple[QuadContourSet,QuadContourSet]:
    #     phi_peak = self.phi_peak.to_value_in(phi_unit)
    #     fnu_peak = self.fnu_peak.to_value_in(fnu_unit)
    #
    #     abgrid = self.parameter_grids
    #     a_wind_grid = abgrid.a_wind_grid
    #     beta_sh_grid = abgrid.beta_sh_grid
    #
    #     if not (
    #         phi_peak.shape
    #         == fnu_peak.shape
    #         == a_wind_grid.shape
    #         == beta_sh_grid.shape
    #     ):
    #         raise ValueError(
    #             "等高線に使用するグリッドのshapeが"
    #             "一致していません. "
    #             f"phi_peak={phi_peak.shape}, "
    #             f"fnu_peak={fnu_peak.shape}, "
    #             f"a_wind={a_wind_grid.shape}, "
    #             f"beta_sh={beta_sh_grid.shape}"
    #         )
    #
    #     contour_a_wind = ax.contour(
    #         phi_peak,
    #         fnu_peak,
    #         a_wind_grid,
    #         levels=(
    #             contourconfs.a_wind.levels
    #             .to_nparray(a_wind_grid)
    #         ),
    #         **contourconfs.a_wind.style.to_kwargs(),
    #     )
    #     ax.clabel(
    #         contour_a_wind,
    #         **contourconfs.a_wind.label.to_kwargs(),
    #     )
    #
    #     contour_beta_sh = ax.contour(
    #         phi_peak,
    #         fnu_peak,
    #         beta_sh_grid,
    #         levels=(
    #             contourconfs.beta_sh.levels
    #             .to_nparray(beta_sh_grid)
    #         ),
    #         **contourconfs.beta_sh.style.to_kwargs(),
    #     )
    #     ax.clabel(
    #         contour_beta_sh,
    #         **contourconfs.beta_sh.label.to_kwargs(),
    #     )
    #
    #     return (
    #             contour_a_wind,
    #             contour_beta_sh,
    #         )
    #
    def plot_phitheta_contour(
        self,
        ax:Axes,
        contourconfs: ContourConfigures,
        *,
        phi_unit: str,
        fnu_unit: str,
    )->tuple[QuadContourSet,QuadContourSet]:
        phi_peak = self.phi_peak.to_value_in(phi_unit)
        fnu_peak = self.fnu_peak.to_value_in(fnu_unit)

        phi_th = self.phi_th.to_value_in(phi_unit)
        theta0 = self.theta0.values

        if not (
            phi_peak.shape
            == fnu_peak.shape
            == phi_th.shape
            == theta0.shape
        ):
            raise ValueError(
                "等高線に使用するグリッドのshapeが"
                "一致していません. "
                f"phi_peak={phi_peak.shape}, "
                f"fnu_peak={fnu_peak.shape}, "
                f"phi_th={phi_th.shape}, "
                f"theta0={theta0.shape}"
            )

        contour_phith = ax.contour(
            phi_peak,
            fnu_peak,
            phi_th,
            levels=(
                contourconfs.a_wind.levels
                .to_nparray(phi_th)
            ),
            **contourconfs.a_wind.style.to_kwargs(),
        )
        ax.clabel(
            contour_phith,
            **contourconfs.a_wind.label.to_kwargs(),
        )

        contour_theta0 = ax.contour(
            phi_peak,
            fnu_peak,
            theta0,
            levels=(
                contourconfs.beta_sh.levels
                .to_nparray(theta0)
            ),
            **contourconfs.beta_sh.style.to_kwargs(),
        )
        ax.clabel(
            contour_theta0,
            **contourconfs.beta_sh.label.to_kwargs(),
        )

        return (
                contour_phith,
                contour_theta0,
            )
    #
    @staticmethod
    def plot_colormap(
        phi_peak:QuantityGrid,
        fnu_peak:QuantityGrid,
        z:QuantityGrid,
        ax: Axes,
        style: ColormapConfigure,
        *,
        phi_unit: str,
        fnu_unit: str,
        z_unit: str
    ):
        phi_val = phi_peak.to_value_in(phi_unit)
        fnu_val = fnu_peak.to_value_in(fnu_unit)
        z_val = z.to_value_in(z_unit)
        return ax.pcolormesh(
            phi_val,fnu_val,z_val,
            shading=style.shading,
            cmap=style.cmap,
            norm=style.norm.type
        )

    def plot_theta_colormap(
        self,
        ax: Axes,
        style: ColormapConfigure,
        *,
        phi_unit: str,
        fnu_unit: str,
    ):
        return self.plot_colormap(
            self.phi_peak,self.fnu_peak,
            z=self.theta,
            ax=ax,
            style=style,
            phi_unit=phi_unit,
            fnu_unit=fnu_unit,
            z_unit=""
        )
    def plot_tau_colormap(
        self,
        ax: Axes,
        style: ColormapConfigure,
        *,
        phi_unit: str,
        fnu_unit: str,
    ):
        return self.plot_colormap(
            self.phi_peak,self.fnu_peak,
            z=self.tau_th,
            ax=ax,
            style=style,
            phi_unit=phi_unit,
            fnu_unit=fnu_unit,
            z_unit=""
        )
    def plot_phi_th_colormap(
        self,
        ax: Axes,
        style: ColormapConfigure,
        *,
        phi_unit: str,
        fnu_unit: str,
    ):
        return self.plot_colormap(
            self.phi_peak,self.fnu_peak,
            z=self.phi_th,
            ax=ax,
            style=style,
            phi_unit=phi_unit,
            fnu_unit=fnu_unit,
            z_unit=phi_unit
        )
    def plot_lnu_th_colormap(
        self,
        ax: Axes,
        style: ColormapConfigure,
        *,
        phi_unit: str,
        fnu_unit: str,
        lnu_unit: str
    ):
        return self.plot_colormap(
            self.phi_peak,self.fnu_peak,
            z=self.lnu_th,
            ax=ax,
            style=style,
            phi_unit=phi_unit,
            fnu_unit=fnu_unit,
            z_unit=lnu_unit
        )

    @staticmethod
    def show_colorbar(
        mesh,
        fig:Figure,
        ax: Axes,
        style: ColormapConfigure
    ):
        label = style.colorbar.label
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label(f"{label}")
    # def plot_metadata(
    #     self,
    #     ax: Axes,
    #     plotconf: PlotConfig
    # ):
    #     strings = []
    #     conf_metadata = plotconf.metadata
    #     conf_eps_b = conf_metadata["eps_b"]
    #     conf_eps_th = conf_metadata["eps_th"]
    #     conf_mu = conf_metadata["mu"]
    #     conf_mu_e = conf_metadata["mu_e"]
    #     strings.append(conf_eps_b.format_value(self.eps_b))
    #     strings.append(conf_eps_th.format_value(self.eps_th))
    #     strings.append(conf_mu.format_value(self.mu))
    #     strings.append(conf_mu_e.format_value(self.mu_e))
    #     ax.set_title(
    #         ", ".join(strings),
    #         fontsize=16
    #     )
    #
    # def select_parameters_in(
    #     self,
    #     observation: ChevalierScatter,
    #     peak_table: LambdaPeakTable,
    #     *,
    #     phi_unit: str,
    #     fnu_unit: str,
    # )->AllowningParameters:
    #     phi_grid = (
    #         self.phi_peak(peak_table)
    #         .to_value_in(phi_unit)
    #     )
    #     fnu_grid = (
    #         self.fnu_peak(peak_table)
    #         .to_value_in(fnu_unit)
    #     )
    #     abgrid = self.parameter_grids
    #
    #     mask = observation.contains(
    #         phi_grid,
    #         fnu_grid,
    #         phi_unit=phi_unit,
    #         fnu_unit=fnu_unit,
    #     )
    #
    #     if mask.shape != abgrid.a_wind_grid.shape:
    #         raise ValueError(
    #             "包含判定マスクとパラメータグリッドの"
    #             "shapeが一致しません. "
    #             f"mask={mask.shape}, "
    #             f"beta_sh={abgrid.a_wind_grid.shape}"
    #         )
    #
    #     phi_theta_q = self.phi_theta
    #     phi_theta = QuantityArray(
    #         phi_theta_q.values[mask],
    #         phi_theta_q.unit
    #     )
    #     lnu_theta_q = self.lnu_theta
    #     lnu_theta = QuantityArray(
    #         lnu_theta_q.values[mask],
    #         lnu_theta_q.unit
    #     )
    #     tau_theta = self.tau_theta[mask]
    #     doppler_delta = models.doppler_delta(abgrid.beta_sh_grid)[mask]
    #
    #     return AllowningParameters(
    #         a_wind = QuantityArray(
    #             values=abgrid.a_wind_grid[mask],
    #             unit=self.a_wind_arr.unit
    #         ),
    #         beta_sh = abgrid.beta_sh_grid[mask],
    #         phi_peak = QuantityArray(
    #             values=phi_grid[mask],
    #             unit=phi_unit
    #         ),
    #         fnu_peak = QuantityArray(
    #             values=fnu_grid[mask],
    #             unit=fnu_unit
    #         ),
    #         tau_theta = tau_theta,
    #         lnu_theta=lnu_theta,
    #         phi_theta=phi_theta,
    #         eps_b=self.eps_b,
    #         eps_th=self.eps_th,
    #         distance=self.distance,
    #         doppler_delta=doppler_delta,
    #         mu=self.mu,
    #         mu_e=self.mu_e
    #     )
