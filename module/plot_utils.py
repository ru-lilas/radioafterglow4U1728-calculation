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
from module.utilities import build_nparray
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from module.utils import FileReader
from module.types import ValueScale,FloatPair

PlotScale = Literal["linear","log"]
AxisDirection = Literal["x","y"]

@dataclass
class LabelConfig:
    prefix: str
    unit: str|None
    fontsize: int

    @property
    def text(self):
        if self.unit is None:
            return f"{self.prefix}"
        else:
            return f"{self.prefix} [{self.unit}]"

@dataclass
class PlotData:
    value: str
    unit: str|None

@dataclass(frozen=True,slots=True)
class AxisConfig:
    scale: ValueScale
    limits: FloatPair|None
    label: LabelConfig
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
    figsize: FloatPair
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

@dataclass
class AxisConfigure:
    ax: Axes
    direction: AxisDirection
    lim: tuple[float,float]
    scale: PlotScale
    ticksize: int
    data: PlotData
    label: LabelConfig

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
class LineStyle:
    color: str = "#000000"
    linewidth: float = 1.0
    linestyle: str = "-"
    alpha: float | None = None
    zorder: float | None = None

    @classmethod
    def from_yaml(
        cls,
        path: Path
    )->Self:
        dict_data = FileReader.yaml_safe(path)
        return from_dict(
            data_class= cls,
            data = dict_data
        )

    def to_kwargs(self):
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None
        }

def build_meshgrid(
    df,
    x_column:str,
    y_column:str,
    z_column:str
):
    pivot = df.pivot(
        index=y_column,
        columns=x_column,
        values=z_column,
    )
    X, Y = np.meshgrid(
        pivot.columns.to_numpy(),
        pivot.index.to_numpy(),
    )
    Z = pivot.to_numpy()
    return (X,Y,Z)

@dataclass
class ColormapConfigure:
    name: str
    over: str|None
    under: str|None

    def build_cmap(self):
        cmap = plt.get_cmap(self.name).copy()
        if (self.over is None) and (self.under is None):
            return cmap
        if self.over is not None:
            cmap.set_over(self.over)
        if self.under is not None:
            cmap.set_under(self.under)
        return cmap

@dataclass
class ContourLevelConfigure:
    scale: PlotScale
    arr: dict[str,float]

    @cached_property
    def levels(self):
        if self.scale == "linear":
            arr = build_nparray.linear(self.arr)
        else:
            arr = build_nparray.log(self.arr)
        return np.concatenate([arr,])

@dataclass
class ContourfConfigure:
    zdata: PlotData
    zscale: PlotScale
    extend: str|None
    levels: NDArray[np.float64]

@dataclass
class ColorbarConfigure:
    cbar: Colorbar
    ticksize: int
    label: LabelConfig

    def show(self):
        self.cbar.set_label(
            self.label.text,
            fontsize = self.label.fontsize
        )
        self.cbar.locator = LogLocator(base=10)
        cbarax = self.cbar.ax
        cbarax.tick_params(labelsize=self.label.fontsize)
        self.cbar.update_ticks()

def fetch_colorbar(conf:dict[str,Any],cbar:Colorbar):
    cbarconf = conf["cbar"]
    cbarlabel = LabelConfig(
        **cbarconf["label"]
    )
    return ColorbarConfigure(
        cbar=cbar,
        ticksize=cbarconf["ticksize"],
        label = cbarlabel
    )

@dataclass
class CurveStyle:
    color: Any
    linestyle: str
    linewidth: float
    label: str|None = None

def curve(
    ax:Axes,
    x,y,
    style:CurveStyle
):
    return ax.plot(
        x,y,
        ls=style.linestyle,
        lw=style.linewidth,
        color=style.color,
        label=style.label
    )[0]

class LightcurvePlotter:
    def plot(self):
        return
    def plot_single(
        self,
        x_axis:AxisConfigure,
        y_axis:AxisConfigure
    ):
        return

