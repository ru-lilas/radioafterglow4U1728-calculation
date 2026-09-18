from matplotlib.collections import QuadMesh
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure
from matplotlib import patches as pltpatch
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, Self
import pandas as pd
from numpy.typing import NDArray
import numpy as np

from module import plot_utils
from module.mydataclasses import (
    FloatGrid,
    FloatGridPair,
    IndexGrid,
    QuantityArray,
    YAMLReadable,
    QuantityData,
    QuantityGrid,
    BoolGrid
)
from module.observation import ChevalierComponent
from module.plot_utils import AxesConfig, AxesRange, AxisConfig, AxisRange, ColorbarConfigure, ColormapConfigure, ColormapPlot, ContourStyle, LegendStyle, MetadataConfig
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
    obs_ranges: plot_utils.RectangleStyles
    thin_boundary: plot_utils.LineStyle
    relativistic_boundary: plot_utils.LineStyle

@dataclass(frozen=True, slots=True)
class OpticallythinBoundaryPlotConfig(YAMLReadable):
    layout: plot_utils.PlotLayoutConfig
    thin_boundary: plot_utils.LineStyle

@dataclass(frozen=True,slots=True)
class ChevalierMetadataConfig:
    fontsize: float
    eps_b:  plot_utils.MetadataConfig
    eps_th: plot_utils.MetadataConfig
    mue_mu: plot_utils.MetadataConfig
    distance: plot_utils.MetadataConfig

@dataclass(frozen=True, slots=True)
class PlotConfig(YAMLReadable):
    layout: plot_utils.PlotLayoutConfig
    contours: ContourConfigures
    styles: Styles
    metadata: ChevalierMetadataConfig
    legend: LegendStyle

@dataclass(frozen=True, slots=True)
class ColormapPlotConfig(YAMLReadable):
    layout: plot_utils.MultiPlotLayoutConfig
    colormaps: ColormapConfigures
    metadata: ChevalierMetadataConfig
    styles: Styles
    legend: LegendStyle

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
        txt_epsb = conf.eps_b.label_format.to_text(self.eps_b)
        txt_epsth = conf.eps_th.label_format.to_text(self.eps_th)
        txt_muemu = conf.mue_mu.label_format.to_text(self.mue_mu)
        txt_distance = conf.distance.label_format.to_text(self.distance.value_in(conf.distance.unit))

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
class RelativisticBoundary:
    phi_boundary:NDArray
    fnu_boundary:NDArray
    style:plot_utils.LineStyle

    @classmethod
    def build(
        cls,
        phi_plot:FloatGrid,
        fnu_plot:FloatGrid,
        theta: FloatGrid,
        theta_crit: float,
        style:plot_utils.LineStyle
    )->Self:
        mask = theta.nearest_values(theta_crit)
        return cls(
            phi_boundary = phi_plot.values[mask.values],
            fnu_boundary = fnu_plot.values[mask.values],
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
class OpticallythinBoundary:
    phi_boundary:NDArray
    fnu_boundary:NDArray
    style:plot_utils.LineStyle

    @classmethod
    def build(
        cls,
        phi_plot:FloatGrid,
        fnu_plot:FloatGrid,
        tau_th : FloatGrid,
        tau_th_crit: float,
        style:plot_utils.LineStyle,
    )->Self:
        mask = tau_th.nearest_values(tau_th_crit)
        return cls(
            phi_boundary = phi_plot.values[mask.values],
            fnu_boundary = fnu_plot.values[mask.values],
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
            r"$\tau_{\Theta}=1$",
            ha="right",
            va="bottom",
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
class ChevalierContour:
    z_quantitygrid: QuantityGrid
    mask_invisible: BoolGrid
    contourconf: plot_utils.ContourConfigure

    @property
    def z_floatgrid(self):
        return FloatGrid(self.z_quantitygrid.to_value_in(self.contourconf.unit))

    @property
    def z_visible(self):
        return self.z_floatgrid.masking(self.mask_invisible)

    def plot(
        self,
        ax: Axes,
        phi_visible: FloatGrid, 
        fnu_visible: FloatGrid
    ):
        conf = self.contourconf
        z_visible = self.z_visible.values
        if not (
            phi_visible.shape
            == fnu_visible.shape
            == z_visible.shape
        ):
            raise ValueError(
                "等高線に使用するグリッドのshapeが"
                "一致していません. "
                f"phi_peak={phi_visible.shape}, "
                f"fnu_peak={fnu_visible.shape}, "
                f"contour={z_visible.shape}, "
            )
        contour = ax.contour(
            phi_visible.values,
            fnu_visible.values,
            z_visible,
            levels = (
                conf.levels
                .levels(z_visible)
            ),
            **conf.style.to_kwargs(),
        )
        quantity_conf = conf.build_quantity_label()
        label_kwargs = conf.label.to_kwargs()
        label_kwargs["fmt"] = quantity_conf.to_text

        ax.clabel(
            contour,
            **label_kwargs
        )
        return contour

@dataclass(frozen=True,slots=True)
class ChevalierPlotRegion:
    phi:FloatGrid
    fnu:FloatGrid
    out_of_plotregion: BoolGrid

    @classmethod
    def build(
        cls,
        phi_qgrid: QuantityGrid,
        fnu_qgrid: QuantityGrid,
        *,
        phi_unit: str,
        fnu_unit: str,
        plotrange: AxesRange
    )->Self:
        phi = FloatGrid(phi_qgrid.to_value_in(phi_unit))
        fnu = FloatGrid(fnu_qgrid.to_value_in(fnu_unit))
        out_of_plotregion = BoolGrid(
            (phi.values <= plotrange.x.min )
            | (phi.values >= plotrange.x.max )
            | (fnu.values <= plotrange.y.min )
            | (fnu.values >= plotrange.y.max )
        )
        return cls(
            phi=phi,
            fnu=fnu,
            out_of_plotregion=out_of_plotregion
        )

    @property
    def phi_in_plotregion(self):
        return self.phi.masking(self.out_of_plotregion)

    @property
    def fnu_in_plotregion(self):
        return self.fnu.masking(self.out_of_plotregion)

@dataclass(frozen=True,slots=True)
class ColorMap:
    x: FloatGrid
    y: FloatGrid
    z: FloatGrid
    cm_style: ColormapConfigure
    cb_style: ColorbarConfigure

    @classmethod
    def build(
        cls,
        xy_grid: ChevalierPlotRegion,
        z_qgrid: QuantityGrid,
        z_mask: BoolGrid,
        cm_style: ColormapConfigure,
        cb_style: ColorbarConfigure
    )->Self:
        z_unit = cm_style.colorbar.label.unit
        z_val = z_qgrid.to_FloatGrid(z_unit)
        return cls(
            xy_grid.phi,xy_grid.fnu,
            z=z_val.masking(z_mask),
            cm_style=cm_style,
            cb_style=cb_style
        )

    def plot(
        self,
        fig:Figure,
        ax:Axes,
    ):
        label = self.cb_style.label
        cm_style = self.cm_style
        cb_style = self.cb_style
        mesh = ax.pcolormesh(
            self.x.values,
            self.y.values,
            self.z.values,
            shading=cm_style.shading,
            norm=cm_style.norm.type,
            cmap=cm_style.cmap
        )
        cbar = fig.colorbar(
            mesh,
            ax=ax,
            pad=cb_style.pad,
            fraction=cb_style.fraction,
        )
        cbar.ax.tick_params(
            labelsize=cb_style.tick_labelsize
        )
        cbar.set_label(
            label.text,
            fontsize=label.fontsize
        )

@dataclass(frozen=True,slots=True)
class BoolianMap:
    x: FloatGrid
    y: FloatGrid
    z: BoolGrid
    cm_style: ColormapConfigure

    @classmethod
    def build(
        cls,
        xy_grid: ChevalierPlotRegion,
        z: BoolGrid,
        cm_style: ColormapConfigure,
    )->Self:
        return cls(
            xy_grid.phi,xy_grid.fnu,
            z=z,
            cm_style=cm_style,
        )

    def plot(
        self,
        fig:Figure,
        ax:Axes,
    ):
        cm_style = self.cm_style
        mesh = ax.pcolormesh(
            self.x.values,
            self.y.values,
            self.z.values.astype(int),
            shading=cm_style.shading,
            norm=cm_style.norm.type,
            cmap=cm_style.cmap
        )
        return mesh

@dataclass(frozen=True,slots=True)
class ChevalierGrid:
    metadata: ChevalierMetadata
    index: IndexGrid
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
    
    @classmethod
    def from_df(
        cls,
        df:pd.DataFrame
    )->Self:
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
            index = IndexGrid.from_df(df),
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

    def mask_SSA_thin(
        self,
        tau_th_crit:float = 1.0
    )->BoolGrid:
        """
            tau_th_crit:float critical value of tau_th
        """
        tau_th = FloatGrid(self.tau_th.values)
        return tau_th.less_than(tau_th_crit)

    def mask_nonrelativistic(
        self,
        theta_crit:float = 1.0
    ):
        return FloatGrid(self.theta.values).less_than(theta_crit)

    def chevalier_plotregion(
        self,
        *,
        x_unit:str,
        y_unit:str,
        plotrange:AxesRange
    )->ChevalierPlotRegion:
        return ChevalierPlotRegion.build(
            phi_qgrid=self.phi_peak,
            fnu_qgrid=self.fnu_peak,
            phi_unit=x_unit,
            fnu_unit=y_unit,
            plotrange=plotrange
        )

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

    def mask_invisible(
        self,
        plot_region: ChevalierPlotRegion
    )->BoolGrid:
        is_ssa_thin = self.mask_SSA_thin()
        is_nonrelativistic = self.mask_nonrelativistic()

        return plot_region.out_of_plotregion.take_or([
            is_ssa_thin,
            is_nonrelativistic
        ])

    def plot_main_contour(
        self,
        ax:Axes,
        plotconf: PlotConfig,
        *,
        phi_unit: str,
        fnu_unit: str,
        plotranges: AxesRange
    )->tuple[QuadContourSet,QuadContourSet]:
        contourconfs = plotconf.contours

        plt_region = ChevalierPlotRegion.build(
            phi_qgrid=self.phi_peak,
            fnu_qgrid=self.fnu_peak,
            phi_unit=phi_unit,
            fnu_unit=fnu_unit,
            plotrange=plotranges
        )
        phi_visible = plt_region.phi_in_plotregion
        fnu_visible = plt_region.fnu_in_plotregion

        invisible = self.mask_invisible(plt_region)

        a_wind_contour = ChevalierContour(
            z_quantitygrid=self.a_wind,
            mask_invisible=invisible,
            contourconf=contourconfs.a_wind
        )
        ca = a_wind_contour.plot(
            ax=ax,
            phi_visible=phi_visible,
            fnu_visible=fnu_visible
        )

        betash_contour = ChevalierContour(
            z_quantitygrid=self.beta_sh,
            mask_invisible=invisible,
            contourconf=contourconfs.beta_sh
        )
        cb = betash_contour.plot(
            ax=ax,
            phi_visible=phi_visible,
            fnu_visible=fnu_visible
        )

        rela_boundary = RelativisticBoundary.build(
            phi_plot=phi_visible,
            fnu_plot=fnu_visible,
            theta = self.theta.to_FloatGrid(""),
            theta_crit = 1.0,
            style=plotconf.styles.relativistic_boundary
        )
        rela_boundary.plot(ax)

        thin_boundary = OpticallythinBoundary.build(
            phi_plot=phi_visible,
            fnu_plot=fnu_visible,
            tau_th = self.tau_th.to_FloatGrid(""),
            tau_th_crit = 1.0,
            style=plotconf.styles.thin_boundary,
        ) 
        thin_boundary.plot(ax)

        return (ca,cb)

    def colormap(
        self,
        plt_region: ChevalierPlotRegion,
        z: QuantityGrid,
        z_mask: BoolGrid,
        cm_style: ColormapConfigure,
        cb_style: ColorbarConfigure
    )->ColorMap:
        return ColorMap.build(
            xy_grid=plt_region,
            z_qgrid=z,
            z_mask = z_mask,
            cm_style=cm_style,
            cb_style=cb_style
        )

    def boolianmap(
        self,
        plt_region: ChevalierPlotRegion,
        z: BoolGrid,
        cm_style: ColormapConfigure,
    )->BoolianMap:
        return BoolianMap.build(
            xy_grid=plt_region,
            z=z,
            cm_style=cm_style,
        )

    def colormaps(
        self,
        cmplots: list[ColormapPlot],
        *,
        phi_unit: str,
        fnu_unit: str,
        plotranges: AxesRange
    )->list[ColorMap]:
        plt_region = ChevalierPlotRegion.build(
            phi_qgrid=self.phi_peak,
            fnu_qgrid=self.fnu_peak,
            phi_unit=phi_unit,
            fnu_unit=fnu_unit,
            plotrange=plotranges
        )
        phi = plt_region.phi
        fnu = plt_region.fnu

        is_ssa_thin = self.mask_SSA_thin()
        is_nonrelativistic = self.mask_nonrelativistic()
        invisible = plt_region.out_of_plotregion.take_or([
            is_ssa_thin,
            is_nonrelativistic
        ])

        colormaps:list[ColorMap] = []
        for cmplot in cmplots:
            style=cmplot.style
            cb = style.colorbar
            colormaps.append(
                ColorMap.build(
                    xy_grid=plt_region,
                    z_qgrid=cmplot.quantity,
                    z_mask = invisible,
                    cm_style=style,
                    cb_style=cb
                )
            )
        return colormaps

    def bool_contain_in_obs(
        self,
        obs: ChevalierComponent,
        *,
        phi_unit: str,
        fnu_unit: str,
    )->BoolGrid:
        chevalier_grids = FloatGridPair(
            x=self.phi_peak.to_FloatGrid(phi_unit),
            y=self.fnu_peak.to_FloatGrid(fnu_unit)
        )
        obs_region = obs.allowing_range(
            x_unit=phi_unit,
            y_unit=fnu_unit
        )
        return chevalier_grids.is_contained_in(obs_region)

    def allowed_modelindex(
        self,
        obs: ChevalierComponent,
        *,
        phi_unit: str,
        fnu_unit: str,
    )->NDArray[np.int64]:
        in_observation_range = self.bool_contain_in_obs(
            obs,
            phi_unit=phi_unit,
            fnu_unit=fnu_unit
        )

        return self.index.select(in_observation_range)

    def df_modelcandidates(
        self,
        df: pd.DataFrame,
        obs: ChevalierComponent,
        phi_unit: str,
        fnu_unit: str,
    )->pd.DataFrame:
        model_indices = self.allowed_modelindex(
            obs,
            phi_unit = phi_unit,
            fnu_unit = fnu_unit,
        )
        df_candidates:pd.DataFrame = df.loc[model_indices]
        df_candidates[ModelColumnNames.NU] = obs.nu.value
        df_candidates.attrs[ModelParameterNames.NU_UNIT] = obs.nu.unit

        return df_candidates
