from dataclasses import dataclass
from functools import cached_property
from typing import Any, TypeAlias, Literal
from matplotlib.ticker import LogLocator
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.tri import TriContourSet
from module import dataframe_utils
from matplotlib.axes import Axes

PlotScale = Literal["linear","log"]
AxisDirection = Literal["x","y"]

@dataclass
class LabelConfigure:
    prefix: str
    unit: str|None
    fontsize: int

    @cached_property
    def labelstr(self):
        if self.unit is None:
            return f"{self.prefix}"
        else:
            return f"{self.prefix} [{self.unit}]"

@dataclass
class AxisArrayData:
    value: str
    unit: str|None

@dataclass
class AxisConfigure:
    ax: Axes
    direction: AxisDirection
    lim: tuple[float,float]
    scale: PlotScale
    ticksize: int
    data: AxisArrayData
    label: LabelConfigure

    def __post_init__(self):
        if self.direction == "x":
            self.ax.set_xlabel(self.label.labelstr,fontsize=self.label.fontsize)
            self.ax.set_xscale(self.scale)
            self.ax.set_xlim(self.lim)
            self.ax.tick_params(axis=self.direction,which="major", pad=20)
        else:
            self.ax.set_ylabel(self.label.labelstr,fontsize=self.label.fontsize)
            self.ax.set_yscale(self.scale)
            self.ax.set_ylim(self.lim)
        self.ax.tick_params(axis=self.direction,labelsize=self.ticksize)
        self.ax.tick_params(axis="both",which="major", length=10, width=1, direction="in")
        self.ax.tick_params(axis="both",which="minor", length=8, width=1, direction="in")

    def array(self,df):
        return dataframe_utils.extract_column_as_ndarray(
            df,self.data.value
        )

@dataclass
class ColorbarConfigure:
    cbar: Colorbar
    ticksize: int
    label: LabelConfigure

    def show(self):
        self.cbar.set_label(
            self.label.labelstr,
            fontsize = self.label.fontsize
        )
        self.cbar.locator = LogLocator(base=10)
        cbarax = self.cbar.ax
        cbarax.tick_params(labelsize=self.label.fontsize)
        self.cbar.update_ticks()

def fetch_colorbar(conf:dict[str,Any],cbar:Colorbar):
    cbarconf = conf["cbar"]
    cbarlabel = LabelConfigure(
        **cbarconf["label"]
    )
    return ColorbarConfigure(
        cbar=cbar,
        ticksize=cbarconf["ticksize"],
        label = cbarlabel
    )
