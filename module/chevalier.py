from matplotlib.collections import QuadMesh
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, Self
import pandas as pd
from numpy.typing import NDArray
import numpy as np

from module import plot_utils
from module.mydataclasses import (
    FloatGrid,
    QuantityArray,
    YAMLReadable,
    QuantityData,
    QuantityGrid,
    BoolGrid
)
from module.observation import ChevalierScatter
from module.plot_utils import AxesConfig, AxesRange, AxisConfig, AxisRange, ColormapConfigure, ColormapPlot, ContourStyle, MetadataConfig
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
    thin_boundary: plot_utils.LineStyle
    relativistic_boundary: plot_utils.LineStyle

@dataclass(frozen=True, slots=True)
class OpticallythinBoundaryPlotConfig(YAMLReadable):
    layout: plot_utils.PlotLayoutConfig
    thin_boundary: plot_utils.LineStyle

@dataclass(frozen=True,slots=True)
class ChevalierMetadataConfig:
    fontsize: float
    eps_b:  plot_utils.QuantityLabel
    eps_th: plot_utils.QuantityLabel
    mue_mu: plot_utils.QuantityLabel
    distance: plot_utils.QuantityLabel

@dataclass(frozen=True, slots=True)
class PlotConfig(YAMLReadable):
    layout: plot_utils.PlotLayoutConfig
    contours: ContourConfigures
    styles: Styles
    metadata: ChevalierMetadataConfig

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

    def plot(
        self,
        ax:Axes,
        conf:ChevalierMetadataConfig
    )->None:
        txt_epsb = conf.eps_b.to_text(self.eps_b)
        txt_epsth = conf.eps_th.to_text(self.eps_th)
        txt_muemu = conf.mue_mu.to_text(self.mue_mu)
        txt_distance = conf.distance.to_text(self.distance.value_in(conf.distance.unit))

        strings = [
            txt_epsb,
            txt_epsth,
            txt_muemu,
            txt_distance
        ]

        ax.set_title(
            ", ".join(strings),
            fontsize=conf.fontsize
        )


@dataclass(frozen=True,slots=True)
class OpticallythinBoundary:
    theta: FloatArray
    phi_peak: QuantityArray
    lnu_peak: QuantityArray
    fnu_peak: QuantityArray

    def plot(
        self,
        ax:Axes,
        style:plot_utils.LineStyle,
        *,
        phi_unit: str,
        fnu_unit: str,
        plotranges: AxesRange
    ):
        xrange = plotranges.x
        yrange = plotranges.y
        x=self.phi_peak.to_value_in(phi_unit)
        y=self.fnu_peak.to_value_in(fnu_unit)
        in_plot_range = (
            (x >= xrange.min)
            & (x <= xrange.max)
            & (y >= yrange.min)
            & (y <= yrange.max)
        )
        x_visible = x[in_plot_range]
        y_visible = y[in_plot_range]

        if len(x_visible) > 0:
            textpos = plot_utils.LineLabel.log10(
                log10x = np.log10(x_visible),
                log10y = np.log10(y_visible)
            )

            ax.plot(x, y, **style.to_kwargs())
            ax.text(
                *textpos,
                r"$\tau_{\Theta}=1$",
                ha="right",
                va="bottom",
                fontsize=32,
                color=style.color
            )

    def plot_theta_dependence(
        self,
        ax:Axes,
        style:plot_utils.LineStyle,
        y:QuantityArray,
        *,
        y_unit: str
    ):
        ax.plot(
            self.theta,y.to_value_in(y_unit),
            **style.to_kwargs()
        )

    def to_df(
        self,
        *,
        phi_unit: str,
        fnu_unit: str
    ):
        df = pd.DataFrame({
            ModelColumnNames.THETA: self.theta,
            ModelColumnNames.PHI_PEAK_THIN: self.phi_peak.to_value_in(phi_unit),
            ModelColumnNames.FNU_PEAK_THIN: self.fnu_peak.to_value_in(fnu_unit)
        })
        return df

    def select_theta_range(
        self,
        df: pd.DataFrame,
        theta_range: tuple[float, float],
        *,
        theta_column: str = "theta",
    ) -> pd.DataFrame:
        theta_min, theta_max = theta_range

        if theta_min > theta_max:
            raise ValueError(
                "thetaの下限が上限を超えています: "
                f"{theta_min} > {theta_max}"
            )

        if theta_column not in df.columns:
            raise KeyError(
                f"列が存在しません: {theta_column}"
            )

        mask = df[theta_column].between(
            theta_min,
            theta_max,
            inclusive="both",
        )

        return df.loc[mask].sort_values(theta_column) 

    def show_row(
        self,
        theta_range: tuple[float,float],
        *,
        phi_unit: str,
        fnu_unit: str
    ):
        df = self.to_df(phi_unit=phi_unit,fnu_unit=fnu_unit)
        df_selected = self.select_theta_range(df,theta_range)
        print(df_selected.to_string())

@dataclass(frozen=True,slots=True)
class RelativisticBoundary:
    phi_boundary:NDArray
    fnu_boundary:NDArray
    style:plot_utils.LineStyle

    @classmethod
    def build(
        cls,
        phi_plot:FloatGrid,
        fnu_plot:FloatGrid,
        is_relativistic: BoolGrid,
        style:plot_utils.LineStyle
    )->Self:
        return cls(
            phi_boundary = phi_plot.values[is_relativistic.values],
            fnu_boundary = fnu_plot.values[is_relativistic.values],
            style=style
        )

    def label(
        self,
        ax:Axes
    ):
        textpos = plot_utils.LineLabel.log10(
            log10x = np.log10(self.phi_boundary),
            log10y = np.log10(self.fnu_boundary)
        )
        ax.text(
            *textpos,
            r"$\Theta=1$",
            ha="left",
            va="top",
            fontsize=32,
            color=self.style.color
        )
        return

    def plot(
        self,
        ax:Axes
    ):
        self.label(ax)
        return ax.plot(
            self.phi_boundary,
            self.fnu_boundary,
            **self.style.to_kwargs()
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
    is_ssa_thin: BoolGrid
    phi_peak_thin: QuantityGrid
    fnu_peak_thin: QuantityGrid
    lnu_peak_thin: QuantityGrid
    
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

        is_ssa_thick = BoolGrid.from_df(
            df=df,
            column=ModelColumnNames.IS_SSA_THICK
        )

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
            is_ssa_thin = ~is_ssa_thick,
            phi_peak_thin = QuantityGrid.from_df(
                df=df,
                unit=phi_unit,
                column=ModelColumnNames.PHI_PEAK_THIN
            ),
            fnu_peak_thin = QuantityGrid.from_df(
                df=df,
                unit=fnu_unit,
                column=ModelColumnNames.FNU_PEAK_THIN
            ),
            lnu_peak_thin = QuantityGrid.from_df(
                df=df,
                unit=lnu_unit,
                column=ModelColumnNames.LNU_PEAK_THIN
            ),
        )

    def build_optically_thin_boundary(
        self
    ):
        phi_peak_thin = self.phi_peak_thin
        fnu_peak_thin = self.fnu_peak_thin
        lnu_peak_thin = self.lnu_peak_thin
        return OpticallythinBoundary(
            theta = np.unique(self.theta.values),
            phi_peak= QuantityArray(
                values = np.unique(phi_peak_thin.values),
                unit = phi_peak_thin.unit
            ),
            lnu_peak = QuantityArray(
                values = np.unique(lnu_peak_thin.values),
                unit = lnu_peak_thin.unit
            ),
            fnu_peak= QuantityArray(
                values = np.unique(fnu_peak_thin.values),
                unit = fnu_peak_thin.unit
            )
        )

    def diagnostic_monotonicity(self):
        phi = FloatGrid(self.phi_peak.values)
        fnu = FloatGrid(self.fnu_peak.values)

        print(
            "phi_peakが非単調な行",
            phi.find_nonmonotonic_lines(axis=1)
        )
        print(
            "phi_peakが非単調な列",
            phi.find_nonmonotonic_lines(axis=0)
        )
        print(
            "fnu_peakが非単調な行",
            fnu.find_nonmonotonic_lines(axis=1)
        )
        print(
            "fnu_peakが非単調な列",
            fnu.find_nonmonotonic_lines(axis=0)
        )
        return

    def turning_point(self):

        turning_points = FloatGrid(self.fnu_peak.values).find_turning_points_in_columns()
        fnu_values = self.fnu_peak.values
        records: list[dict[str, object]] = []

        nrows = self.fnu_peak.values.shape[0]

        for turning_row, column in turning_points:
            row_start = max(0, turning_row - 1)
            row_stop = min(nrows, turning_row + 2)

            for row in range(row_start, row_stop):
                records.append({
                    "turning_row": turning_row,
                    "row": row,
                    "column": column,
                    ModelColumnNames.A_WIND: self.a_wind.values[row, column],
                    ModelColumnNames.BETA_SH: self.beta_sh.values[row, column],
                    "theta": self.theta.values[row, column],
                    "tau_theta": self.tau_th.values[row, column],
                    "fnu_peak": self.fnu_peak.values[row, column],
                    "is_ssa_thin": self.is_ssa_thin.values[
                        row,
                        column,
                    ],
                })

        turning_neighborhood_df = pd.DataFrame(records)
        print(turning_neighborhood_df)
        turning_neighborhood_df["tau_theta_lt_1"] = (
            turning_neighborhood_df["tau_theta"] < 1.0
        )
        print(
            turning_neighborhood_df[
                [
                    "column",
                    "theta",
                    "tau_theta",
                    "tau_theta_lt_1",
                    "is_ssa_thin",
                    ModelColumnNames.FNU_PEAK
                ]
            ].to_string(index=False)
        )

    def mask_relativistic(
        self,
        theta_crit:float = 1.0
    )->BoolGrid:
        """
            theta_crit: critical value of theta for relativistic
        """
        theta = self.theta.values
        row_indices = np.argmin(
            np.abs(theta - theta_crit),
            axis=0
        )
        column_indices = np.arange(theta.shape[1])
        mask = np.zeros_like(
            theta,
            dtype=bool,
        )

        mask[row_indices, column_indices] = True

        return BoolGrid(mask)

    def mask_outof_plotrange(
        self,
        phi_grid:FloatGrid,
        fnu_grid:FloatGrid,
        plotranges: AxesRange
    )->BoolGrid:
        return BoolGrid(
            (phi_grid.values <= plotranges.x.min )
            | (phi_grid.values >= plotranges.x.max )
            | (fnu_grid.values <= plotranges.y.min )
            | (fnu_grid.values >= plotranges.y.max )
        )

    def plot_relativistic_boundary(
        self,
        ax:Axes,
        phi_grid:FloatGrid,
        fnu_grid:FloatGrid,
        is_relativistic: BoolGrid,
        style:plot_utils.LineStyle,
    ):
        phi_boundary = phi_grid.values[is_relativistic.values]
        fnu_boundary = fnu_grid.values[is_relativistic.values]
        ax.plot(
            phi_boundary,
            fnu_boundary,
            **style.to_kwargs()
        )
        textpos = plot_utils.LineLabel.log10(
            log10x = np.log10(phi_boundary),
            log10y = np.log10(fnu_boundary)
        )

        ax.text(
            *textpos,
            r"$\Theta=1$",
            ha="left",
            va="top",
            fontsize=32,
            color=style.color
        )
        return

    def plot_main_contour(
        self,
        ax:Axes,
        plotconf: PlotConfig,
        *,
        phi_unit: str,
        fnu_unit: str,
        plotranges: AxesRange
    )->tuple[QuadContourSet,QuadContourSet]:
        phi_peak = FloatGrid(self.phi_peak.to_value_in(phi_unit))
        fnu_peak = FloatGrid(self.fnu_peak.to_value_in(fnu_unit))

        contourconfs = plotconf.contours
        a_wind_unit = contourconfs.a_wind.unit
        beta_unit = contourconfs.beta_sh.unit
        a_wind_grid = FloatGrid(self.a_wind.to_value_in(a_wind_unit))
        beta_sh_grid = FloatGrid(self.beta_sh.to_value_in(beta_unit))

        out_of_plotrange = self.mask_outof_plotrange(
            phi_grid=phi_peak,
            fnu_grid=fnu_peak,
            plotranges=plotranges
        )

        phi_masked = phi_peak.masking(out_of_plotrange)
        fnu_masked = fnu_peak.masking(out_of_plotrange)
        a_masked = a_wind_grid.masking(out_of_plotrange)
        b_masked = beta_sh_grid.masking(out_of_plotrange)

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
                contourconfs.a_wind.levels
                .to_nparray(a_masked.values)
            ),
            **contourconfs.a_wind.style.to_kwargs(),
        )
        quantity_conf = contourconfs.a_wind.build_quantity_label()
        label_kwargs = contourconfs.a_wind.label.to_kwargs()
        label_kwargs["fmt"] = quantity_conf.to_text
        ax.clabel(
            contour_a_wind,
            **label_kwargs
        )

        contour_beta_sh = ax.contour(
            phi_masked.values,
            fnu_masked.values,
            b_masked.values,
            levels=(
                contourconfs.beta_sh.levels
                .to_nparray(b_masked.values)
            ),
            **contourconfs.beta_sh.style.to_kwargs(),
        )
        quantity_conf = contourconfs.beta_sh.build_quantity_label()
        label_kwargs = contourconfs.beta_sh.label.to_kwargs()
        label_kwargs["fmt"] = quantity_conf.to_text
        ax.clabel(
            contour_beta_sh,
            **label_kwargs
        )

        rela_boundary = RelativisticBoundary.build(
            phi_plot=phi_masked,
            fnu_plot=fnu_masked,
            is_relativistic=self.mask_relativistic(),
            style=plotconf.styles.relativistic_boundary
        )

        rela_boundary.plot(ax)

        return (
                contour_a_wind,
                contour_beta_sh,
            )

    def plot_colormaps(
        self,
        fig:Figure,
        axes,
        cmplots: list[ColormapPlot],
        *,
        phi_unit: str,
        fnu_unit: str,
        plotranges: AxesRange
    )->None:
        mask = self.is_ssa_thin
        phi = self.phi_peak.to_FloatGrid(phi_unit)
        fnu = self.fnu_peak.to_FloatGrid(fnu_unit)

        out_plot_range = BoolGrid(
            (phi.values <= plotranges.x.min )
            | (phi.values >= plotranges.x.max )
            | (fnu.values <= plotranges.y.min )
            | (fnu.values >= plotranges.y.max )
        )

        for ax, cmplot in zip(axes.flat, cmplots):
            style=cmplot.style
            cb = style.colorbar
            label = cb.label
            z_val = FloatGrid(cmplot.quantity.to_value_in(label.unit))
            z_plot = z_val.masking(mask)
            mesh = ax.pcolormesh(
                phi.values,
                fnu.values,
                z_plot.values,
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


@dataclass(frozen=True,slots=True)
class ChevalierColormap:
    phi: FloatGrid
    fnu: FloatGrid
    ssa_mask: BoolGrid
    cmplots:list[ColormapPlot]

    def plot_colormaps(
        self,
        fig:Figure,
        axes,
    )->None:

        for ax, cmplot in zip(axes.flat, self.cmplots):
            style=cmplot.style
            cb = style.colorbar
            label = cb.label
            z_val = FloatGrid(cmplot.quantity.to_value_in(label.unit))
            z_plot = z_val.masking(self.ssa_mask)
            mesh = ax.pcolormesh(
                self.phi.values,
                self.fnu.values,
                z_plot.values,
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
