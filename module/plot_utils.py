from dataclasses import asdict, dataclass
from functools import cached_property
from typing import Any, Literal, Self
from dacite import from_dict
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import LogLocator
from matplotlib.colorbar import Colorbar
from numpy.typing import NDArray
from module import dataframe_utils
from module.mydataclasses import QuantityGrid, ValueArray, YAMLReadable
from module.utilities import build_nparray
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from module.utils import FileReader
from module.types import FloatArray, ValueScale,FloatPair

PlotScale = Literal["linear","log"]
AxisDirection = Literal["x","y"]

def format_value(
    value: float,
    prefix: str,
    fmt: str,
)->str:
    formatted = f"{value:{fmt}}"

    if fmt.endswith("e"):
        mantissa,exponent = formatted.split("e")
        exponent = int(exponent)
        return (
            rf"${prefix}"
            rf"={mantissa}\times10^{{{exponent}}}$"
        )
    else:
        return rf"${prefix}={formatted}$"

@dataclass
class LabelConfigure:
    prefix: str
    unit: str
    fontsize: int

    @property
    def text(self):
        if self.unit == "":
            return f"{self.prefix}"
        else:
            return f"{self.prefix} [{self.unit}]"

@dataclass
class PlotData:
    value: str
    unit: str|None

@dataclass(frozen=True, slots=True)
class SubplotConfig:
    nrows: int
    ncols: int
    sharex: bool = False
    sharey: bool = False
    hide_shared_labels: bool = True

@dataclass(frozen=True,slots=True)
class AxisConfig:
    scale: ValueScale
    limits: FloatPair|None
    label: LabelConfigure
    major_ticks: tuple[float, ...] | None = None
    minor_ticks: tuple[float, ...] | None = None

@dataclass(frozen=True, slots=True)
class TickStyle:
    labelsize: float
    direction: str = "in"
    major_length: float = 10.0
    minor_length: float = 8.0
    width: float = 1.0
    pad: float = 5.0

@dataclass(frozen=True, slots=True)
class AxesConfig:
    x: AxisConfig
    y: AxisConfig
    ticks: TickStyle

@dataclass(frozen=True, slots=True)
class FigureConfig:
    figsize: tuple[float,float]
    dpi: int = 300
    layout_engine: str = "constrained"

@dataclass(frozen=True, slots=True)
class PlotLayoutConfig:
    figure: FigureConfig
    axes: AxesConfig

    @classmethod
    def from_yaml(
        cls,
        path: Path
    )->Self:
        dict_data = FileReader.yaml_safe(path)
        return from_dict(
            data_class = cls,
            data = dict_data
        )

@dataclass(frozen=True, slots=True)
class MultiPlotLayoutConfig:
    figure: FigureConfig
    axes: AxesConfig
    subplots: SubplotConfig

class AxesConfigurator:
    @staticmethod
    def apply_axis(
        ax: Axes,
        direction: AxisDirection,
        config: AxisConfig
    )->None:
        if direction == "x":
            ax.set_xscale(config.scale)
            ax.set_xlabel(
                config.label.text,
                fontsize=config.label.fontsize,
            )

            if config.limits is not None:
                ax.set_xlim(config.limits)

            if config.major_ticks is not None:
                ax.set_xticks(
                    config.major_ticks,
                    minor=False,
                )

            if config.minor_ticks is not None:
                ax.set_xticks(
                    config.minor_ticks,
                    minor=True,
                )

        else:
            ax.set_yscale(config.scale)
            ax.set_ylabel(
                config.label.text,
                fontsize=config.label.fontsize,
            )

            if config.limits is not None:
                ax.set_ylim(config.limits)

            if config.major_ticks is not None:
                ax.set_yticks(
                    config.major_ticks,
                    minor=False,
                )

            if config.minor_ticks is not None:
                ax.set_yticks(
                    config.minor_ticks,
                    minor=True,
                )

    @staticmethod
    def apply_tick_style(
        ax: Axes,
        style: TickStyle
    )->None:
        ax.tick_params(
            axis="both",
            which="major",
            length=style.major_length,
            width=style.width,
            direction=style.direction,
            labelsize=style.labelsize,
            pad=style.pad,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            length=style.minor_length,
            width=style.width,
            direction=style.direction,
        )

    @classmethod
    def apply(
        cls,
        ax: Axes,
        config: AxesConfig,
    )->None:
        cls.apply_axis(ax,"x",config.x)
        cls.apply_axis(ax,"y",config.y)
        cls.apply_tick_style(ax,config.ticks)

def create_configured_axes(
    config: PlotLayoutConfig,
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(
        figsize=config.figure.figsize,
        dpi=config.figure.dpi,
    )
    AxesConfigurator.apply(ax,config.axes)

    return fig, ax

def create_configured_subplots(
    config: MultiPlotLayoutConfig,
) -> tuple[Figure, NDArray]:
    fig, axes = plt.subplots(
        nrows=config.subplots.nrows,
        ncols=config.subplots.ncols,
        figsize=config.figure.figsize,
        dpi=config.figure.dpi,
        sharex=config.subplots.sharex,
        sharey=config.subplots.sharey,
        layout=config.figure.layout_engine,
        squeeze=False,
    )

    for ax in axes.flat:
        AxesConfigurator.apply(
            ax,
            config.axes,
        )

    if config.subplots.hide_shared_labels:
        if config.subplots.sharex:
            for ax in axes[:-1, :].flat:
                ax.set_xlabel("")

        if config.subplots.sharey:
            for ax in axes[:, 1:].flat:
                ax.set_ylabel("")

    return fig, axes

@dataclass
class AxisConfigure:
    ax: Axes
    direction: AxisDirection
    lim: tuple[float,float]
    scale: PlotScale
    ticksize: int
    data: PlotData
    label: LabelConfigure

    def __post_init__(self):
        if self.direction == "x":
            self.ax.set_xlabel(self.label.text,fontsize=self.label.fontsize)
            self.ax.set_xscale(self.scale)
            self.ax.set_xlim(self.lim)
            self.ax.tick_params(axis=self.direction,which="major", pad=20)
        else:
            self.ax.set_ylabel(self.label.text,fontsize=self.label.fontsize)
            self.ax.set_yscale(self.scale)
            self.ax.set_ylim(self.lim)
        self.ax.tick_params(axis=self.direction,labelsize=self.ticksize)
        self.ax.tick_params(axis="both",which="major", length=10, width=1, direction="in")
        self.ax.tick_params(axis="both",which="minor", length=8, width=1, direction="in")

    def array(self,df):
        return dataframe_utils.extract_column_as_ndarray(
            df,self.data.value
        )

@dataclass(frozen=True, slots=True)
class Style:
    def to_kwargs(self)->dict[str,Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None
        }

@dataclass(frozen=True, slots=True)
class LineStyle(YAMLReadable):
    color: str = "#000000"
    linewidth: float = 1.0
    linestyle: str = "-"
    alpha: float | None = None
    zorder: float | None = None

    def to_kwargs(self):
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None
        }

@dataclass(frozen=True, slots=True)
class ScatterStyle:
    marker: str = "o"
    markersize: float = 6.0
    markerfacecolor: str = "#000000"
    markeredgecolor: str = "#000000"
    markeredgewidth: float = 1.0
    ecolor: str = "#000000"
    elinewidth: float = 1.0
    capsize: float = 0.0
    capthick: float = 1.0
    linestyle: str = "none"
    alpha: float | None = None
    zorder: float | None = None

    def to_kwargs(self)->dict[str,Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None
        }

@dataclass(frozen=True, slots=True)
class RectangleStyle(Style):
    facecolor: str
    edgecolor: str
    linewidth: float = 1.0
    alpha: float = 0.0
    zorder: float|None = None

@dataclass(frozen=True, slots=True)
class ContourLevelConfigure:
    start: float|None
    stop: float|None
    num: int

    def to_nparray(self,grid:FloatArray):
        if self.start is None:
            start_compensated:float = grid.min()
        else:
            start_compensated = self.start
    
        if self.stop is None:
            stop_compensated:float = grid.max()
        else:
            stop_compensated = self.stop

        return np.geomspace(
            start=start_compensated,
            stop=stop_compensated,
            num=self.num
        )

@dataclass(frozen=True,slots=True)
class ContourStyle(Style):
    colors: str = "#000000"
    linestyles: str = "-"

@dataclass(frozen=True,slots=True)
class ContourLabelStyle(Style):
    fontsize: int = 10
    fmt: str = "%.1e"

@dataclass(frozen=True,slots=True)
class ContourConfigure:
    style: ContourStyle
    levels: ContourLevelConfigure
    label: ContourLabelStyle
    unit: str

@dataclass(frozen=True, slots=True)
class ColorbarConfigure:
    show: bool
    pad: float
    fraction: float
    tick_labelsize: float
    label: LabelConfigure

@dataclass(frozen=True, slots=True)
class NormConfig:
    type: str

@dataclass(frozen=True, slots=True)
class ColormapConfigure(YAMLReadable):
    cmap: str
    shading: Literal["flat","nearest","gouraud","auto"]
    vmin: float | None
    vmax: float | None
    norm: NormConfig
    colorbar: ColorbarConfigure

@dataclass(frozen=True, slots=True)
class ColormapPlot:
    quantity: QuantityGrid
    style: ColormapConfigure

@dataclass(frozen=True, slots=True)
class MetadataConfig:
    show: bool
    prefix: str
    fmt: str
    unit: str|None

    def format_value(
        self,
        value:float
    )->str:
        return format_value(
            value=value,
            prefix=self.prefix,
            fmt=self.fmt
        )
