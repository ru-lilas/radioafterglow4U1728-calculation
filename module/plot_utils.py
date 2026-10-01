from dataclasses import asdict, dataclass, field
from functools import cached_property
from typing import Any, Literal, Self, Sequence, TypeVar
from cattrs import Converter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize,ListedColormap
from matplotlib.container import ErrorbarContainer
from matplotlib.contour import QuadContourSet
from matplotlib.artist import Artist
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, StepPatch
from matplotlib.text import Text
from matplotlib.ticker import LogLocator
from matplotlib import colormaps
from matplotlib.colorbar import Colorbar
from numpy.typing import ArrayLike, NDArray
from module.mydataclasses import FloatGrid, NDArraySpace, QuantityData, QuantityGrid, ValueArray, YAMLReadable
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from module.utils import FileReader
from module.types import FloatArray, ValueScale,FloatPair

AxisDirection = Literal["x","y"]
ComponentType = Literal[
    "line",
    "step",
    "errorbar",
    "contour",
    "rectangle",
    "text",
    "legend",
]

StyleValue = TypeVar("StyleValue")

converter = Converter()

bool_cmap = ListedColormap(
    [
        (0.0, 0.0, 0.0, 0.0),  # False: 透明
        (0.12, 0.47, 0.71, 0.5),  # True
    ],
    name="boolian",
)
colormaps.register(bool_cmap)

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

@dataclass(frozen=True, slots=True)
class QuantityLabel:
    prefix: str
    fmt: str
    unit: str
    suffix: str = ""
    show_equals: bool = True

    def formatted_value(
        self,
        value: float,
    ) -> str:
        formatted = f"{value:{self.fmt}}"

        if self.fmt.endswith("e"):
            mantissa, exponent = formatted.split("e")

            return (
                rf"{mantissa}"
                rf"\times10^{{{int(exponent)}}}"
            )

        return formatted

    def to_text(
            self,
            value: float,
        ) -> str:
            formatted = self.formatted_value(value)
            separator = ("=" if self.show_equals else r"\,") if self.prefix else ""
            unit = rf"\,{self.unit}" if self.unit else ""
            suffix = rf"\,{self.suffix}" if self.suffix else ""
            return rf"${self.prefix}{separator}{formatted}{unit}{suffix}$"

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

@dataclass(frozen=True, slots=True)
class SubplotConfig:
    nrows: int = 1
    ncols: int = 1
    sharex: bool = False
    sharey: bool = False
    hide_shared_labels: bool = True

    def __post_init__(self) -> None:
        if self.nrows < 1 or self.ncols < 1:
            raise ValueError(
                "nrowsとncolsは1以上でなければなりません: "
                f"nrows={self.nrows}, ncols={self.ncols}"
            )

    @property
    def size(self) -> int:
        return self.nrows * self.ncols

@dataclass(frozen=True,slots=True)
class AxisRange:
    min: float
    max: float

@dataclass(frozen=True,slots=True)
class AxesRange:
    x: AxisRange
    y: AxisRange

@dataclass(frozen=True,slots=True)
class AxisConfig:
    scale: ValueScale
    limits: FloatPair|None
    label: LabelConfigure
    major_ticks: tuple[float, ...] | None = None
    minor_ticks: tuple[float, ...] | None = None

    def range(
        self
    )->AxisRange:
        lim = self.limits
        if lim is not None:
            return AxisRange(
                min=lim[0],
                max=lim[1]
            )
        else:
            raise ValueError(
                "描画範囲が設定されていないため, 値を取得できません."
            )

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

    def range(
        self
    )->AxesRange:
        return AxesRange(
            x=self.x.range(),
            y=self.y.range()
        )

    @property
    def x_unit(self):
        return self.x.label.unit

    @property
    def y_unit(self):
        return self.y.label.unit

@dataclass(frozen=True, slots=True)
class FigureConfig:
    figsize: tuple[float,float]
    dpi: int = 300
    layout_engine: str = "constrained"

@dataclass(frozen=True, slots=True)
class PlotLayout:
    figure: FigureConfig
    axes: AxesConfig
    subplots: SubplotConfig = field(
        default_factory=SubplotConfig
    )

    @classmethod
    def from_yaml(
        cls,
        path: Path
    )->Self:
        dict_data = FileReader.yaml_safe(path)
        return converter.structure(
            dict_data,
            cls,
        )

    def create(
        self,
    ) -> tuple[Figure, NDArray]:
        figure_config = self.figure
        subplot_config = self.subplots

        fig, axes = plt.subplots(
            nrows=subplot_config.nrows,
            ncols=subplot_config.ncols,
            figsize=figure_config.figsize,
            dpi=figure_config.dpi,
            sharex=subplot_config.sharex,
            sharey=subplot_config.sharey,
            layout=figure_config.layout_engine,
            squeeze=False,
        )

        for ax in axes.flat:
            AxesConfigurator.apply(
                ax,
                self.axes,
            )

        if subplot_config.hide_shared_labels:
            if subplot_config.sharex:
                for ax in axes[:-1, :].flat:
                    ax.set_xlabel("")

            if subplot_config.sharey:
                for ax in axes[:, 1:].flat:
                    ax.set_ylabel("")

        return fig, axes

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

@dataclass(frozen=True, slots=True)
class Style:
    def to_kwargs(self)->dict[str,Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None
        }

@dataclass(frozen=True, slots=True)
class LineStyle(Style):
    color: str
    linewidth: float = 1.0
    linestyle: str = "-"
    alpha: float | None = None
    zorder: float | None = None

@dataclass(frozen=True, slots=True)
class TextStyle(Style):
    color: str
    fontsize: float = 12.0
    ha: str = "left"
    va: str = "top"

def select_style_value(
    values: Sequence[StyleValue],
    index: int,
) -> StyleValue:
    if len(values) == 0:
        raise ValueError(
            "styleの候補が設定されていません."
        )

    if len(values) == 1:
        return values[0]

    try:
        return values[index]
    except IndexError as error:
        raise IndexError(
            f"style index={index}に対応する値がありません: "
            f"len={len(values)}"
        ) from error

@dataclass(frozen=True, slots=True)
class LineStyleConfig(YAMLReadable):
    colors: list[str]
    linewidth: float = 1.0
    linestyle: str = "-"
    alpha: float | None = None
    zorder: float | None = None

    def resolve(
        self,
        index: int,
    )->LineStyle:
        return LineStyle(
            color=select_style_value(
                self.colors,
                index,
            ),
            linewidth=self.linewidth,
            linestyle=self.linestyle,
            alpha=self.alpha,
            zorder=self.zorder,
        )

@dataclass(frozen=True, slots=True)
class ErrorbarStyle(Style):
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

@dataclass(frozen=True, slots=True)
class ErrorbarStyleConfig(YAMLReadable):
    colors: list[str]
    marker: str = "o"
    markersize: float = 6.0
    markeredgewidth: float = 1.0
    elinewidth: float = 1.0
    capsize: float = 0.0
    capthick: float = 1.0
    linestyle: str = "none"
    alpha: float | None = None
    zorder: float | None = None

    def resolve(
        self,
        index: int,
    )->ErrorbarStyle:
        color = select_style_value(
            self.colors,
            index,
        )
        return ErrorbarStyle(
            marker=self.marker,
            markersize=self.markersize,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=self.markeredgewidth,
            ecolor=color,
            elinewidth=self.elinewidth,
            capsize=self.capsize,
            capthick=self.capthick,
            linestyle=self.linestyle,
            alpha=self.alpha,
            zorder=self.zorder,
        )

@dataclass(frozen=True, slots=True)
class RectangleStyle(Style):
    facecolor: str
    edgecolor: str = "#000000"
    linewidth: float = 1.0
    alpha: float = 1.0
    zorder: float | None = None
    hatch: str | None = None

@dataclass(frozen=True, slots=True)
class RectangleStyleConfig(YAMLReadable):
    facecolors: list[str]
    hatches: list[str | None]
    edgecolor: str = "#000000"
    linewidth: float = 1.0
    alpha: float = 1.0
    zorder: float | None = None

    def resolve(
        self,
        index: int,
    ) -> RectangleStyle:
        return RectangleStyle(
            facecolor=select_style_value(
                self.facecolors,
                index,
            ),
            edgecolor=self.edgecolor,
            linewidth=self.linewidth,
            alpha=self.alpha,
            zorder=self.zorder,
            hatch=select_style_value(
                self.hatches,
                index,
            ),
        )

@dataclass(frozen=True, slots=True)
class ContourLevelConfig:
    start: float|None = None
    stop: float|None = None
    num: int|None = None
    scale: Literal["linear","log"] = "log"
    values: list[float] | None = None

    def start_value(
        self,
        grid:NDArray[np.float64]
    )->float:
        return (
            grid.min()
            if self.start is None
            else self.start
        )

    def stop_value(
        self,
        grid:NDArray[np.float64]
    )->float:
        return (
            grid.max()
            if self.stop is None
            else self.stop
        )

    def build_levels_auto(
        self,
        grid_values:NDArray[np.float64]
    )->NDArray[np.float64]:
        if self.num is None:
            raise ValueError(
                "valuesを指定しない場合は"
                "numが必要です"
            )
        start = self.start_value(grid_values)
        stop = self.stop_value(grid_values)
        space = NDArraySpace(
            start,
            stop,
            self.num
        )

        if self.scale=="linear":
            levels = space.linspace() 
        elif self.scale=="log":
            levels = space.geomspace()
        else:
            raise ValueError(
                f"未対応のscaleが設定されています: {self.scale}"
            )

        return levels

    @staticmethod
    def grid_values(
        grid:FloatArray
    )->NDArray[np.float64]:
        grid_values = np.asarray(
            np.ma.compressed(grid),
            dtype=np.float64,
        )
        grid_values = grid_values[
            np.isfinite(grid_values)
        ]
        if grid_values.size==0:
            raise ValueError(
                "等高線levelを決められません"
            )
        else:
            return grid_values

    def levels(self,grid:FloatArray):
        grid_values = self.grid_values(grid)
        if self.values is not None:
            levels = np.asarray(
                self.values,
                dtype=np.float64
            )
        else:
            levels = self.build_levels_auto(
                grid_values
            )

        levels = np.unique(levels)

        # gridの範囲外にあるlevelを除外
        in_grid_range = (
            (grid_values.min() <= levels)
            & (levels <= grid_values.max())
        )
        levels = levels[in_grid_range]

        if levels.size == 0:
            raise ValueError(
                "gridの範囲内に等高線levelがありません: "
                f"grid=[{grid_values.min()}, "
                f"{grid_values.max()}]"
            )

        return levels

@dataclass(frozen=True,slots=True)
class ContourLineStyle(Style):
    colors: str = "#000000"
    linestyles: str = "-"

@dataclass(frozen=True,slots=True)
class ContourLabelStyle(Style):
    inline: bool
    inline_spacing: int
    fontsize: int

@dataclass(frozen=True,slots=True)
class ContourStyleConfig(YAMLReadable):
    unit: str
    lines: ContourLineStyle
    levels: ContourLevelConfig
    label: ContourLabelStyle
    label_format: QuantityLabel

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
    unit: str
    label_format:QuantityLabel

@dataclass(frozen=True, slots=True)
class MetadataGroupConfig:
    fontsize: float
    items: dict[str,MetadataConfig]

@dataclass(frozen=True, slots=True)
class LegendStyle(Style):
    fontsize: int
    loc: str = "best"

@dataclass(frozen=True, slots=True)
class ComponentConfig:
    type: ComponentType
    style: (
        LineStyleConfig
        | ErrorbarStyleConfig
        | ContourStyleConfig
        | RectangleStyleConfig
        | TextStyle
        | LegendStyle
    )
    show_legend: bool = False
    label_format: QuantityLabel | None = None

    def __post_init__(self) -> None:
        if self.type in ("line", "step"):
            if not isinstance(
                self.style,
                LineStyleConfig,
            ):
                raise TypeError(
                    f"type={self.type!r}には"
                    "LineStyleConfigが必要です: "
                    f"{type(self.style).__name__}"
                )

        elif self.type == "errorbar":
            if not isinstance(
                self.style,
                ErrorbarStyleConfig,
            ):
                raise TypeError(
                    "type='errorbar'には"
                    "ErrorbarStyleConfigが必要です: "
                    f"{type(self.style).__name__}"
                )

        elif self.type == "contour":
            if not isinstance(
                self.style,
                ContourStyleConfig,
            ):
                raise TypeError(
                    "type='contour'には"
                    "ContourStyleConfigが必要です: "
                    f"{type(self.style).__name__}"
                )

        elif self.type == "rectangle":
            if not isinstance(
                self.style,
                RectangleStyleConfig,
            ):
                raise TypeError(
                    "type='rectangle'には"
                    "RectangleStyleConfigが必要です: "
                    f"{type(self.style).__name__}"
                )

        elif self.type == "text":
            if not isinstance(self.style, TextStyle):
                raise TypeError(
                    "type='text'にはTextStyleが必要です: "
                    f"{type(self.style).__name__}"
                )

        elif self.type == "legend":
            if not isinstance(self.style, LegendStyle):
                raise TypeError(
                    "type='legend'にはLegendStyleが必要です: "
                    f"{type(self.style).__name__}"
                )

        if self.show_legend and self.label_format is None:
            raise ValueError(
                "legend=trueのcomponentにはlabel_formatが必要です."
            )

    @property
    def line_style_config(
        self,
    ) -> LineStyleConfig:
        if self.type not in ("line", "step"):
            raise TypeError(
                "line_style_configへアクセスできるのは"
                "type='line'または'step'の場合だけです: "
                f"type={self.type!r}"
            )

        if not isinstance(
            self.style,
            LineStyleConfig,
        ):
            raise TypeError(
                "styleがLineStyleConfigではありません: "
                f"{type(self.style).__name__}"
            )

        return self.style

    @property
    def errorbar_style_config(
        self,
    ) -> ErrorbarStyleConfig:
        if self.type != "errorbar":
            raise TypeError(
                "errorbar_style_configへアクセスできるのは"
                "type='errorbar'の場合だけです: "
                f"type={self.type!r}"
            )

        if not isinstance(
            self.style,
            ErrorbarStyleConfig,
        ):
            raise TypeError(
                "styleがErrorbarStyleConfigではありません: "
                f"{type(self.style).__name__}"
            )

        return self.style

    @property
    def contour_style_config(
        self,
    ) -> ContourStyleConfig:
        if self.type != "contour":
            raise TypeError(
                "contour_style_configへアクセスできるのは"
                "type='contour'の場合だけです: "
                f"type={self.type!r}"
            )

        if not isinstance(
            self.style,
            ContourStyleConfig,
        ):
            raise TypeError(
                "styleがContourStyleConfigではありません: "
                f"{type(self.style).__name__}"
            )

        return self.style

    @property
    def rectangle_style_config(
        self,
    ) -> RectangleStyleConfig:
        if self.type != "rectangle":
            raise TypeError(
                "rectangle_style_configへアクセスできるのは"
                "type='rectangle'の場合だけです: "
                f"type={self.type!r}"
            )

        if not isinstance(
            self.style,
            RectangleStyleConfig,
        ):
            raise TypeError(
                "styleがRectangleStyleConfigではありません: "
                f"{type(self.style).__name__}"
            )

        return self.style

    @property
    def text_style_config(self) -> TextStyle:
        if self.type != "text" or not isinstance(self.style, TextStyle):
            raise TypeError(
                "text_style_configへアクセスできるのは"
                "type='text'の場合だけです."
            )
        return self.style

    @property
    def legend_style_config(self) -> LegendStyle:
        if self.type != "legend" or not isinstance(self.style, LegendStyle):
            raise TypeError(
                "legend_style_configへアクセスできるのは"
                "type='legend'の場合だけです."
            )
        return self.style

    def legend_label(self, value: float | None = None) -> str:
        if self.label_format is None:
            raise ValueError("legend用label_formatが設定されていません.")
        if value is None:
            return self.label_format.prefix
        return self.label_format.to_text(value)

    def resolve_line_style(
        self,
        index: int,
    ) -> LineStyle:
        return self.line_style_config.resolve(index)

    def resolve_errorbar_style(
        self,
        index: int,
    ) -> ErrorbarStyle:
        return self.errorbar_style_config.resolve(index)

    def resolve_rectangle_style(
        self,
        index: int,
    ) -> RectangleStyle:
        return self.rectangle_style_config.resolve(index)

def structure_component_config(
    data: dict[str, Any],
    _: Any,
)->ComponentConfig:
    component_type = data["type"]

    if component_type in ("line", "step"):
        style_type = LineStyleConfig
    elif component_type == "errorbar":
        style_type = ErrorbarStyleConfig
    elif component_type == "contour":
        style_type = ContourStyleConfig
    elif component_type == "rectangle":
        style_type = RectangleStyleConfig
    elif component_type == "text":
        style_type = TextStyle
    elif component_type == "legend":
        style_type = LegendStyle
    else:
        raise ValueError(
            "未対応のcomponent typeです: "
            f"{component_type}"
        )

    return ComponentConfig(
        type=component_type,
        style=converter.structure(
            data["style"],
            style_type,
        ),
        show_legend=data.get("legend", False),
        label_format=(
            converter.structure(data["label_format"], QuantityLabel)
            if data.get("label_format") is not None
            else None
        ),
    )

converter.register_structure_hook(
    ComponentConfig,
    structure_component_config,
)

def structure_components(
    data: dict[str, Any],
    _: Any,
) -> dict[str, ComponentConfig]:
    return {
        name: (
            ComponentConfig(
                type="legend",
                style=converter.structure(component, LegendStyle),
            )
            if name == "legend"
            else converter.structure(component, ComponentConfig)
        )
        for name, component in data.items()
    }

@dataclass(frozen=True, slots=True)
class PlotConfig:
    layout: PlotLayout
    metadata: MetadataGroupConfig
    components: dict[str, ComponentConfig]

    @classmethod
    def from_yaml(
        cls,
        path: Path,
    )->Self:
        data = FileReader.yaml_safe(path)
        return cls(
            layout=converter.structure(data["layout"], PlotLayout),
            metadata=converter.structure(data["metadata"], MetadataGroupConfig),
            components=structure_components(data["components"], dict),
        )

@dataclass(frozen=True, slots=True)
class MetadataComponent:
    quantity: QuantityData
    config: MetadataConfig

    def add_strings(
        self,
        strings:list[str]
    ):
        config = self.config
        if not config.show:
            return None
        else:
            value = self.quantity.value_in(config.unit)
            text = config.label_format.to_text(value)
            strings.append(text)
            return None

class LineLabel:
    @staticmethod
    def log10(
        log10x: NDArray[np.float64],
        log10y: NDArray[np.float64]
    )->tuple[float,float]:
        log10x_center = 0.5 * (log10x.min() + log10x.max())
        i = np.argmin(np.abs(log10x - log10x_center))
        return (
            10**log10x[i],
            10**log10y[i]
        )

@dataclass(frozen=True, slots=True)
class LegendEntry:
    component: ComponentConfig
    artist: Artist
    value: float | None = None

@dataclass(frozen=True, slots=True)
class Plotter:
    ax: Axes

    def text(
        self,
        x: float,
        y: float,
        label: str,
        style: TextStyle,
        *,
        coordinates: Literal["data", "axes"] = "data",
    ) -> Text:
        transform = (
            self.ax.transAxes
            if coordinates == "axes"
            else self.ax.transData
        )
        return self.ax.text(
            x, y, label,
            transform=transform,
            **style.to_kwargs(),
        )

    def legend(
        self,
        entries: Sequence[LegendEntry],
        config: ComponentConfig,
    ) -> Legend | None:
        selected = [entry for entry in entries if entry.component.show_legend]
        if not selected:
            return None
        return self.ax.legend(
            handles=[entry.artist for entry in selected],
            labels=[
                entry.component.legend_label(entry.value)
                for entry in selected
            ],
            **config.legend_style_config.to_kwargs(),
        )

    def line(
        self,
        x:ArrayLike,
        y:ArrayLike,
        style: LineStyle
    )->list[Line2D]:
        return self.ax.plot(
            x,y,
            **style.to_kwargs()
        )

    def step(
        self,
        values: ArrayLike,
        edges: ArrayLike,
        style: LineStyle,
        *,
        baseline:float = 0.0
    )->StepPatch:
        return self.ax.stairs(
            values,edges,
            baseline=baseline,
            **style.to_kwargs(),
        )

    def errorbar(
        self,
        x: ArrayLike,
        y: ArrayLike,
        yerr: ArrayLike,
        style: ErrorbarStyle
    )->ErrorbarContainer:
        return self.ax.errorbar(
            x,y,
            yerr=yerr,
            **style.to_kwargs(),
        )

    def contour(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: FloatArray,
        style: ContourStyleConfig,
    )->QuadContourSet:
        return self.ax.contour(
            x,
            y,
            z,
            levels=style.levels.levels(z),
            **style.lines.to_kwargs(),
        )

    def contour_labels(
        self,
        contour_set: QuadContourSet,
        style: ContourStyleConfig,
    )->list[Text]:
        return self.ax.clabel(
            contour_set,
            fmt=style.label_format.to_text,
            **style.label.to_kwargs(),
        )

    def rectangle(
        self,
        xy: tuple[float, float],
        width: float,
        height: float,
        style: RectangleStyle,
    ) -> Rectangle:
        patch = Rectangle(
            xy,
            width,
            height,
            **style.to_kwargs(),
        )
        self.ax.add_patch(patch)
        return patch

    def title(
        self,
        strings: list[str],
        fontsize: float
    ):
        self.ax.set_title(
            ", ".join(strings),
            fontsize=fontsize
        )
