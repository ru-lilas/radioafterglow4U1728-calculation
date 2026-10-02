from matplotlib.collections import QuadMesh
from matplotlib.container import ErrorbarContainer
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
from module import observation
from module.inputs_as_dataclass import CandidateModelsSelection
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
from module.plot_utils import AxesConfig, AxesRange, AxisConfig, AxisRange, ColorbarConfigure, ColormapConfigure, ColormapPlot,LegendStyle, MetadataConfig
from module.types import FloatArray, FloatPair
from module import models
from matplotlib.axes import Axes

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
class PlotRegion:
    x:FloatGrid
    y:FloatGrid
    out_of_plotregion: BoolGrid

    @classmethod
    def build(
        cls,
        x_grid_quantity: QuantityGrid,
        y_grid_quantity: QuantityGrid,
        *,
        x_unit: str,
        y_unit: str,
        plotrange: AxesRange
    )->Self:
        x = FloatGrid(x_grid_quantity.to_value_in(x_unit))
        y = FloatGrid(y_grid_quantity.to_value_in(y_unit))
        out_of_plotregion = BoolGrid(
            (x.values <= plotrange.x.min )
            | (x.values >= plotrange.x.max )
            | (y.values <= plotrange.y.min )
            | (y.values >= plotrange.y.max )
        )
        return cls(
            x,y,
            out_of_plotregion
        )

    @property
    def x_in_plotregion(self):
        return self.x.masking(self.out_of_plotregion)

    @property
    def y_in_plotregion(self):
        return self.y.masking(self.out_of_plotregion)


@dataclass(frozen=True,slots=True)
class BoundaryLabel:
    phi_boundary: NDArray
    fnu_boundary: NDArray
    theta_crit: float
    style: plot_utils.TextStyle

    def plot(self, plotter: plot_utils.Plotter) -> None:
        if self.phi_boundary.size == 0:
            return None

        textpos = plot_utils.LineLabel.log10(
            log10x=np.log10(self.phi_boundary),
            log10y=np.log10(self.fnu_boundary),
        )
        plotter.text(
            *textpos,
            rf"$\Theta={self.theta_crit:.0f}$",
            self.style,
        )

@dataclass(frozen=True,slots=True)
class RelativisticBoundary:
    phi_boundary:NDArray
    fnu_boundary:NDArray
    theta_crit: float
    style:plot_utils.LineStyle

    @classmethod
    def build(
        cls,
        plotregion: PlotRegion,
        theta: FloatGrid,
        theta_crit: float,
        style:plot_utils.LineStyle
    )->Self:
        phi = plotregion.x.values
        fnu = plotregion.y.values
        mask = (
            theta.nearest_values(theta_crit).values
            & ~plotregion.out_of_plotregion.values
            & np.isfinite(phi)
            & np.isfinite(fnu)
            & (phi > 0)
            & (fnu > 0)
        )
        return cls(
            phi_boundary = phi[mask],
            fnu_boundary = fnu[mask],
            style=style,
            theta_crit = theta_crit
        )

    def plot(
        self,
        plotter:plot_utils.Plotter
    ):
        if self.phi_boundary.size == 0:
            return None
        plotter.line(self.phi_boundary,self.fnu_boundary,self.style)

    def build_label(
        self,
        style:plot_utils.TextStyle
    )->BoundaryLabel:
        return BoundaryLabel(
            self.phi_boundary,
            self.fnu_boundary,
            self.theta_crit,
            style
        )


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
class ChevalierContours:
    plotregion: PlotRegion
    z_grid: FloatGrid
    style: plot_utils.ContourStyleConfig

    @classmethod
    def build(
        cls,
        plotregion:PlotRegion,
        z_grid_quantity: QuantityGrid,
        style:plot_utils.ContourStyleConfig,
        mask:BoolGrid
    )->Self:
        z_grid = z_grid_quantity.to_FloatGrid(style.unit)
        return cls(
            plotregion = plotregion,
            z_grid=z_grid.masking(mask),
            style=style
        )

    def plot(self,plotter:plot_utils.Plotter):
        pr = self.plotregion
        style = self.style
        ctr = plotter.contour(
            x=pr.x_in_plotregion.values,
            y=pr.y_in_plotregion.values,
            z=self.z_grid.values,
            style=style
        )
        plotter.contour_labels(ctr,style)

@dataclass(frozen=True,slots=True)
class ObservedRegion:
    x: float
    y: float
    width: float
    height: float
    style: plot_utils.RectangleStyle

    @classmethod
    def build(
        cls,
        x:float,
        y:float,
        *,
        x_err:float,
        y_err:float,
        style:plot_utils.RectangleStyle
    )->Self:
        return cls(
            x = x - x_err,
            y = y - y_err,
            width = 2.0 * x_err,
            height = 2.0 * y_err,
            style = style
        )

    def plot(
        self,
        plotter:plot_utils.Plotter
    )->pltpatch.Rectangle:
        return plotter.rectangle(
            xy=(self.x,self.y),
            width=self.width,
            height=self.height,
            style=self.style
        )

@dataclass(frozen=True,slots=True)
class BestfitPoint:
    x: float
    y: float
    style: plot_utils.ErrorbarStyle

    @classmethod
    def build(
        cls,
        x:float,
        y:float,
        *,
        x_err:float,
        y_err:float,
        style:plot_utils.ErrorbarStyle
    )->Self:
        return cls(
            x = x - x_err,
            y = y - y_err,
            style = style
        )

    def plot(
        self,
        plotter:plot_utils.Plotter
    )->ErrorbarContainer:
        return plotter.errorbar(
            x=self.x,
            y=self.y,
            yerr=0,
            style=self.style
        )

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
class PeakGrid(models.SynchrotronPeak):

    def boolgrid_contain_in(
        self,
        obs_peak_range: ChevalierComponent,
        *,
        phi_unit: str = models.DefaultUnits.phi,
        fnu_unit: str = models.DefaultUnits.fnu,
    )->BoolGrid:
        chevalier_grids = FloatGridPair(
            x=self.phi_grid.to_FloatGrid(phi_unit),
            y=self.fnu_grid.to_FloatGrid(fnu_unit)
        )
        obs_region = obs_peak_range.allowing_range(
            x_unit=phi_unit,
            y_unit=fnu_unit
        )
        return chevalier_grids.is_contained_in(obs_region)

    def indices_of_contained_models(
        self,
        boolgrid: BoolGrid
    )->NDArray[np.int64]:
        return self.index.select(boolgrid)

    def reduce_to_plotregion(
        self,
        plotrange:AxesRange,
        *,
        phi_unit:str = models.DefaultUnits.phi,
        fnu_unit:str = models.DefaultUnits.fnu
    )->PlotRegion:
        return PlotRegion.build(
            x_grid_quantity=self.phi_grid,
            y_grid_quantity=self.fnu_grid,
            plotrange=plotrange,
            x_unit=phi_unit,
            y_unit=fnu_unit
        )

@dataclass(frozen=True,slots=True)
class CandidateModels:
    model_index: NDArray[np.int64]
    nu: QuantityData

def build_candidate_longdf(
    df_peaks:pd.DataFrame,
    df_obs_long: observation.LongformatLightcurve,
    selection_config: CandidateModelsSelection
)->pd.DataFrame:
    peakgrid = PeakGrid.from_df(df_peaks)

    selection_freqs = selection_config.observation_frequencies.to_QuantityData_list()
    dfs_obs_nu = [
        df_obs_long.select_frequency(nu,unit=nu.unit)
        for nu in selection_freqs
    ]

    dfs_candidates:list = []
    for i,df_obs_nu in enumerate(dfs_obs_nu):
        obs_peak_range = df_obs_nu.peak

        is_contained = peakgrid.boolgrid_contain_in(obs_peak_range)
        model_indices = peakgrid.indices_of_contained_models(is_contained)

        df_candidates:pd.DataFrame = df_peaks.loc[model_indices].copy()
        df_candidates[models.ColumnNames.NU] = obs_peak_range.nu.value
        df_candidates.attrs[models.MetadataNames.NU_UNIT] = obs_peak_range.nu.unit

        dfs_candidates.append(
            df_candidates
        )
    longdf = pd.concat(
        dfs_candidates,
        axis="index"
    )
    columns = [
        models.ColumnNames.MODEL_INDEX,
        models.ColumnNames.NU
    ]
    longdf = longdf[columns]
    longdf.attrs = {
        models.MetadataNames.NU_UNIT:
            longdf.attrs[models.MetadataNames.NU_UNIT]
    }
    if not isinstance(longdf,pd.DataFrame):
        raise TypeError(
            f"longdfの型が不正です:{type(longdf)}"
        )
    return longdf
