from matplotlib.collections import QuadMesh
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, Self

from numpy.typing import NDArray
import numpy as np

from module import plot_utils
from module.mydataclasses import (
    FloatGrid,
    YAMLReadable,
    QuantityData,
    QuantityGrid,
    BoolGrid
)
from module.observation import ChevalierScatter
from module.plot_utils import AxesConfig, AxisConfig, ColormapConfigure, ColormapPlot, ContourStyle, MetadataConfig
from module.types import FloatArray, FloatPair
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
    is_ssa_thick: BoolGrid
    
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

        n_false = df["is_ssa_thick"].eq(False).sum()
        false_rows = df.loc[df["is_ssa_thick"].eq(False)]
        print(false_rows)

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
            ),
            is_ssa_thick = BoolGrid.from_df(
                df=df,
                column=ModelColumnNames.IS_SSA_THICK
            )
        )

    @staticmethod
    def plot_range(
        axisconfig:AxisConfig
    )->tuple[float,float]:
        lim = axisconfig.limits
        if lim is not None:
            return lim
        else:
            raise ValueError("描画範囲が設定されていないため, 値を取得できません")

    def grid_masked(
        self,
        grid_val:FloatArray
    )->FloatArray:
        return np.ma.masked_where(
            ~self.is_ssa_thick.values.reshape(grid_val.shape),
            grid_val,
        )

    def plot_main_contour(
        self,
        ax:Axes,
        confs: ContourConfigures,
        *,
        phi_unit: str,
        fnu_unit: str,
        phi_range: FloatPair,
        fnu_range: FloatPair
    )->tuple[QuadContourSet,QuadContourSet]:
        phi_peak = FloatGrid(self.phi_peak.to_value_in(phi_unit))
        fnu_peak = FloatGrid(self.fnu_peak.to_value_in(fnu_unit))

        a_wind_unit = confs.a_wind.unit
        beta_unit = confs.beta_sh.unit
        a_wind_grid = FloatGrid(self.a_wind.to_value_in(a_wind_unit))
        beta_sh_grid = FloatGrid(self.beta_sh.to_value_in(beta_unit))

        out_plot_range = BoolGrid(
            (phi_peak.values <= phi_range[0] )
            | (phi_peak.values >= phi_range[1])
            | (fnu_peak.values <= fnu_range[0])
            | (fnu_peak.values >= fnu_range[1])
        )

        phi_masked = phi_peak.masking(out_plot_range)
        fnu_masked = fnu_peak.masking(out_plot_range)
        a_masked = a_wind_grid.masking(out_plot_range)
        b_masked = beta_sh_grid.masking(out_plot_range)

        if not (
            phi_masked.shape
            == fnu_masked.shape
            == a_masked.shape
            == b_masked.shape
        ):
            raise ValueError(
                "等高線に使用するグリッドのshapeが"
                "一致していません. "
                f"phi_peak={phi_masked.shape}, "
                f"fnu_peak={fnu_masked.shape}, "
                f"a_wind={a_masked.shape}, "
                f"beta_sh={b_masked.shape}"
            )

        contour_a_wind = ax.contour(
            phi_masked.values,
            fnu_masked.values,
            a_masked.values,
            levels=(
                confs.a_wind.levels
                .to_nparray(a_masked.values)
            ),
            **confs.a_wind.style.to_kwargs(),
        )
        ax.clabel(
            contour_a_wind,
            **confs.a_wind.label.to_kwargs(),
        )

        contour_beta_sh = ax.contour(
            phi_masked.values,
            fnu_masked.values,
            b_masked.values,
            levels=(
                confs.beta_sh.levels
                .to_nparray(b_masked.values)
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
            z_val = cmplot.quantity.to_value_in(label.unit)
            z_masked = self.grid_masked(z_val)
            mesh = ax.pcolormesh(
                phi_peak_val,
                fnu_peak_val,
                z_masked,
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
