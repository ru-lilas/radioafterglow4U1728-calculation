from matplotlib.collections import QuadMesh
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, Self

from module import plot_utils
from module.mydataclasses import (
    YAMLReadable,
    QuantityData,
    QuantityGrid
)
from module.observation import ChevalierScatter
from module.plot_utils import ColormapConfigure, ColormapPlot, ContourStyle, MetadataConfig
from module.types import FloatArray
from module.parameter_table import ModelParameterNames,ModelColumnNames
from enum import StrEnum, auto
from matplotlib.axes import Axes
from module.compute_lightcurve import AllowningParameters
from module.utils import FileReader

@dataclass(frozen=True,slots=True)
class ContourConfigures:
    a_wind: plot_utils.ContourConfigure
    beta_sh: plot_utils.ContourConfigure

@dataclass(frozen=True,slots=True)
class ColormapConfigures:
    theta: plot_utils.ColormapConfigure
    tau_th: plot_utils.ColormapConfigure
    phi_th: plot_utils.ColormapConfigure
    lnu_th: plot_utils.ColormapConfigure
    beta_sh: plot_utils.ColormapConfigure
    a_syn: plot_utils.ColormapConfigure
    a_wind: plot_utils.ColormapConfigure

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
    beta_sh: QuantityGrid
    a_syn: QuantityGrid
    a_wind: QuantityGrid
    
    @classmethod
    def from_csv(
        cls,
        path:Path
    )->Self:
        df = FileReader.table_from_csv(path,idx="index")
        metadata = ChevalierMetadata.from_attrs(df.attrs)
        phi_unit = metadata.phi_unit
        fnu_unit = metadata.fnu_unit
        lnu_unit = metadata.lnu_unit
        a_wind_unit = metadata.a_wind_unit

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
            ),
            beta_sh = QuantityGrid.from_df(
                df=df,
                unit="",
                column=ModelColumnNames.BETA_SH
            ),
            a_syn = QuantityGrid.from_df(
                df=df,
                unit=a_wind_unit,
                column=ModelColumnNames.A_SYN
            ),
            a_wind = QuantityGrid.from_df(
                df=df,
                unit=a_wind_unit,
                column=ModelColumnNames.A_WIND
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
    def plot_main_contour(
        self,
        ax:Axes,
        confs: ContourConfigures,
        *,
        phi_unit: str,
        fnu_unit: str,
    )->tuple[QuadContourSet,QuadContourSet]:
        phi_peak = self.phi_peak.to_value_in(phi_unit)
        fnu_peak = self.fnu_peak.to_value_in(fnu_unit)

        a_wind_unit = confs.a_wind.unit
        beta_unit = confs.beta_sh.unit
        a_wind_grid = self.a_wind.to_value_in(a_wind_unit)
        beta_sh_grid = self.beta_sh.to_value_in(beta_unit)

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
                confs.a_wind.levels
                .to_nparray(a_wind_grid)
            ),
            **confs.a_wind.style.to_kwargs(),
        )
        ax.clabel(
            contour_a_wind,
            **confs.a_wind.label.to_kwargs(),
        )

        contour_beta_sh = ax.contour(
            phi_peak,
            fnu_peak,
            beta_sh_grid,
            levels=(
                confs.beta_sh.levels
                .to_nparray(beta_sh_grid)
            ),
            **confs.beta_sh.style.to_kwargs(),
        )
        ax.clabel(
            contour_beta_sh,
            **confs.beta_sh.label.to_kwargs(),
        )

        return (
                contour_a_wind,
                contour_beta_sh,
            )

    def plot_colormap(
        self,
        z:QuantityGrid,
        ax: Axes,
        style: ColormapConfigure,
        *,
        phi_unit: str,
        fnu_unit: str,
    )->QuadMesh:
        z_unit = style.colorbar.label.unit
        return ax.pcolormesh(
            self.phi_peak.to_value_in(phi_unit),
            self.fnu_peak.to_value_in(fnu_unit),
            z.to_value_in(z_unit),
            shading=style.shading,
            cmap=style.cmap,
            norm=style.norm.type,
        )
    def plot_colormaps(
        self,
        fig:Figure,
        axes,
        cmplots: list[ColormapPlot],
        *,
        phi_unit: str,
        fnu_unit: str,
    )->None:
        phi_peak_val:FloatArray = self.phi_peak.to_value_in(phi_unit)
        fnu_peak_val:FloatArray = self.fnu_peak.to_value_in(fnu_unit)

        for ax, cmplot in zip(axes.flat, cmplots):
            style=cmplot.style
            cb = style.colorbar
            label = cb.label
            mesh = ax.pcolormesh(
                phi_peak_val,
                fnu_peak_val,
                cmplot.quantity.to_value_in(label.unit),
                shading=style.shading,
                norm=style.norm.type,
                cmap=style.cmap
            )
            cbar = fig.colorbar(
                mesh,
                ax=ax,
                pad=cb.pad,
                fraction=cb.fraction,
            )
            cbar.ax.tick_params(
                labelsize=cb.tick_labelsize
            )
            cbar.set_label(
                label.text,
                fontsize=label.fontsize
            )

    @staticmethod
    def show_colorbar(
        mesh,
        fig:Figure,
        ax: Axes,
        style: ColormapConfigure
    ):
        label = style.colorbar.label.text
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
