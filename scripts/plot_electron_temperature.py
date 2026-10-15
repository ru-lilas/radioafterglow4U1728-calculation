"""Compare the GP98 and hot electron temperatures using typed curve settings"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from module import plot_utils as pu
from module.models import DimensionlessElectronTemperature

TemperatureModel = Literal["gp98", "hot"]
SamplingMode = Literal["linear", "log"]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GridConfig:
    lower: float
    upper: float
    samples: int
    sampling: SamplingMode = "log"


@dataclass(frozen=True, slots=True)
class CurveConfig:
    temperature_model: TemperatureModel
    label: str
    style: pu.LineStyle


@dataclass(frozen=True, slots=True)
class ErrorCurveConfig:
    reference_model: TemperatureModel
    approximation_model: TemperatureModel
    label: str
    style: pu.LineStyle


@dataclass(frozen=True, slots=True)
class ReferenceLineConfig:
    percent: float
    label: str
    style: pu.LineStyle


DisplayItem = CurveConfig | ErrorCurveConfig | ReferenceLineConfig


@dataclass(frozen=True, slots=True)
class PanelConfig:
    curves: tuple[DisplayItem, ...]
    axes: pu.AxesConfig


@dataclass(frozen=True, slots=True)
class PlotSettings:
    grid: GridConfig
    panels: tuple[PanelConfig, PanelConfig]
    layout: pu.PlotLayout
    legend: pu.ComponentConfig


@dataclass(frozen=True, slots=True)
class ResolvedCurve:
    config: DisplayItem
    theta: FloatArray


@dataclass(frozen=True, slots=True)
class ResolvedPanel:
    config: PanelConfig
    curves: tuple[ResolvedCurve, ...]


@dataclass(frozen=True, slots=True)
class PlotData:
    theta0: FloatArray
    panels: tuple[ResolvedPanel, ...]


def build_plot_config() -> PlotSettings:
    """Edit sampling, curve instances, ordering and figure appearance here"""
    grid = GridConfig(lower=1e-1, upper=1e2, samples=512, sampling="log")
    gp98 = CurveConfig(
        temperature_model="gp98", label="GP98",
        style=pu.LineStyle(
            color="#000000",
            linewidth=2
        ),
    )
    hot = CurveConfig(
        temperature_model="hot", label=r"Hot: $\Theta=\Theta_0/3$",
        style=pu.LineStyle(
            color="#000000",
            linewidth=1,
            linestyle="--"
        ),
    )
    temperature_panel = PanelConfig(
        curves=(gp98, hot),
        axes=pu.AxesConfig(
            x=pu.AxisConfig("log", None, pu.LabelConfigure(r"$\Theta_0$", "", 20)),
            y=pu.AxisConfig("log", None, pu.LabelConfigure(r"$\Theta$", "", 20)),
            ticks=pu.TickStyle(labelsize=14, major_length=7, minor_length=4, pad=6),
        ),
    )
    error = ErrorCurveConfig(
        reference_model="gp98", approximation_model="hot",
        label="Hot relative to GP98",
        style=pu.LineStyle(color="#000000", linewidth=2),
    )
    one_percent = ReferenceLineConfig(
        percent=10.0, label="10% error",
        style=pu.LineStyle(color="#777777", linewidth=1.5, linestyle=":"),
    )
    error_panel = PanelConfig(
        curves=(error, one_percent),
        axes=replace(temperature_panel.axes,
            y=pu.AxisConfig("linear", None, pu.LabelConfigure("Relative error", "%", 16))),
    )
    # Change only this tuple to move complete panels
    panels = (temperature_panel, error_panel)
    return PlotSettings(
        grid=grid, panels=panels,
        layout=pu.PlotLayout(
            figure=pu.FigureConfig((6, 8), dpi=150, layout_engine="constrained"),
            axes=layout_axes(panels), subplots=pu.SubplotConfig(nrows=2, ncols=1, sharex=True, sharey=False),
        ),
        legend=pu.ComponentConfig(type="legend", style=pu.LegendStyle(fontsize=12, loc="best")),
    )


def real_scalar(value: object, label: str) -> float:
    """Require a finite real scalar before converting it to float"""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real scalar")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def real_vector(value: object, label: str) -> FloatArray:
    """Require a real finite vector before conversion to float64"""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{label} must be a NumPy array")
    if value.ndim != 1 or value.size < 2:
        raise ValueError(f"{label} must be a one-dimensional array with at least two values")
    if value.dtype.kind not in "fiu":
        raise TypeError(f"{label} must contain real numeric values")
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be finite")
    return result


def positive_vector(value: object, label: str) -> FloatArray:
    """Require a positive finite temperature vector"""
    result = real_vector(value, label)
    if np.any(result <= 0):
        raise ValueError(f"{label} must be positive")
    return result


def error_vector(value: object) -> FloatArray:
    """Allow zero but reject negative or nonfinite percentage errors"""
    result = real_vector(value, "Relative error")
    if np.any(result < 0):
        raise ValueError("Relative error must be nonnegative")
    return result


def generate_theta0(config: GridConfig) -> FloatArray:
    """Generate a validated strictly increasing temperature grid"""
    lower = real_scalar(config.lower, "theta0 lower bound")
    upper = real_scalar(config.upper, "theta0 upper bound")
    if not 0 < lower < upper:
        raise ValueError("Require 0 < theta0 lower bound < upper bound")
    samples: object = config.samples
    if isinstance(samples, np.integer):
        samples = samples.item()
    if isinstance(samples, bool) or not isinstance(samples, int):
        raise TypeError("Sample count must be an integer")
    if samples < 2:
        raise ValueError("Sample count must be at least two")
    if config.sampling == "log":
        values = np.geomspace(lower, upper, samples, dtype=np.float64)
    elif config.sampling == "linear":
        values = np.linspace(lower, upper, samples, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported sampling mode: {config.sampling}")
    theta0 = positive_vector(values, "theta0")
    if np.any(np.diff(theta0) <= 0):
        raise ValueError("theta0 samples must be strictly increasing")
    return theta0


def evaluate_temperature(model: TemperatureModel, theta0: FloatArray) -> FloatArray:
    """Evaluate the selected conversion using the existing model implementation"""
    if model == "gp98":
        raw: object = DimensionlessElectronTemperature.gp98(theta0)
    elif model == "hot":
        raw = DimensionlessElectronTemperature.hot(theta0)
    else:
        raise ValueError(f"Unsupported temperature model: {model}")
    theta = positive_vector(raw, f"{model} theta")
    if theta.shape != theta0.shape:
        raise ValueError(f"{model} theta shape differs from theta0")
    return theta


def layout_axes(panels: tuple[PanelConfig, PanelConfig]) -> pu.AxesConfig:
    """Derive shared horizontal and per-row vertical settings from the panels"""
    return pu.AxesConfig(
        x=pu.AxesConfig._single_axis(panels[0].axes.x),
        y=tuple(pu.AxesConfig._single_axis(panel.axes.y) for panel in panels),
        ticks=panels[0].axes.ticks,
    )


def validate_settings(config: PlotSettings) -> None:
    """Require two independent panels with compatible shared horizontal axes"""
    subplots = config.layout.subplots
    if (subplots.nrows, subplots.ncols) != (2, 1) or len(config.panels) != 2:
        raise ValueError("Configure exactly two panels in a two-row, one-column layout")
    if subplots.sharex not in (True, "all", "col") or subplots.sharey not in (False, "none"):
        raise ValueError("Panels must share x and retain independent y axes")
    kinds = [validate_panel(panel) for panel in config.panels]
    if set(kinds) != {"temperature", "error"}:
        raise ValueError("Configure one temperature panel and one error panel")
    first_x = pu.AxesConfig._single_axis(config.panels[0].axes.x)
    if any(pu.AxesConfig._single_axis(panel.axes.x) != first_x for panel in config.panels):
        raise ValueError("Shared x-axis settings must agree across panels")


def validate_panel(panel: PanelConfig) -> Literal["temperature", "error"]:
    """Validate a panel's unique items and distinguish its physical quantity"""
    if not panel.curves:
        raise ValueError("Configure at least one curve per panel")
    selectors: set[str] = set()
    has_temperature = False
    has_error = False
    thresholds: set[float] = set()
    for curve in panel.curves:
        if isinstance(curve, CurveConfig):
            has_temperature = True
            key = curve.temperature_model
            if key not in ("gp98", "hot"):
                raise ValueError(f"Unsupported temperature model: {key}")
        elif isinstance(curve, ErrorCurveConfig):
            has_error = True
            if curve.reference_model not in ("gp98", "hot") or curve.approximation_model not in ("gp98", "hot"):
                raise ValueError("Unsupported error comparison model")
            key = f"error:{curve.approximation_model}:{curve.reference_model}"
        elif isinstance(curve, ReferenceLineConfig):
            threshold = real_scalar(curve.percent, "Error threshold")
            if threshold <= 0 or threshold in thresholds:
                raise ValueError("Error thresholds must be positive and unique")
            thresholds.add(threshold)
            key = f"threshold:{threshold}"
        else:
            raise TypeError("Unsupported panel item")
        if key in selectors:
            raise ValueError(f"Duplicate panel item: {key}")
        selectors.add(key)
        if not isinstance(curve.label, str) or not curve.label.strip():
            raise ValueError("Every panel item requires a nonempty label")
    if has_temperature and (has_error or thresholds):
        raise ValueError("Temperature and percentage items require separate panel configurations")
    if not has_temperature and not has_error:
        raise ValueError("The error panel requires an error curve")
    pu.AxesConfig._single_axis(panel.axes.x)
    pu.AxesConfig._single_axis(panel.axes.y)

    return "temperature" if has_temperature else "error"


def validate_axis(values: FloatArray, axis: pu.AxisConfig, label: str) -> None:
    """Reject incompatible log scales and limits that hide sampled values"""
    if axis.scale == "log" and np.any(values <= 0):
        raise ValueError(f"{label} must be positive on a logarithmic axis; select a linear scale to display zero error")
    if axis.limits is None:
        return
    lower = real_scalar(axis.limits[0], f"{label} axis lower limit")
    upper = real_scalar(axis.limits[1], f"{label} axis upper limit")
    if lower >= upper or (axis.scale == "log" and lower <= 0):
        raise ValueError(f"Invalid {label} axis limits")
    if np.any((values < lower) | (values > upper)):
        raise ValueError(f"Configured {label} limits would hide sampled points")


def validate_data(data: PlotData, config: PlotSettings) -> None:
    """Validate resolved curves against their configured panel before drawing"""
    validate_settings(config)
    theta0 = positive_vector(data.theta0, "theta0")
    if np.any(np.diff(theta0) <= 0):
        raise ValueError("theta0 samples must be strictly increasing")
    if tuple(panel.config for panel in data.panels) != config.panels:
        raise ValueError("Resolved panels disagree with the configured panel order")
    for panel in data.panels:
        if tuple(curve.config for curve in panel.curves) != panel.config.curves:
            raise ValueError("Resolved curves disagree with their panel order")
        validate_axis(theta0, pu.AxesConfig._single_axis(panel.config.axes.x), "theta0")
        for curve in panel.curves:
            values = (positive_vector(curve.theta, curve.config.label) if isinstance(curve.config, CurveConfig)
                      else error_vector(curve.theta))
            if values.shape != theta0.shape:
                raise ValueError("Resolved curve and theta0 shapes differ")
            validate_axis(values, pu.AxesConfig._single_axis(panel.config.axes.y), curve.config.label)


def cached_temperature(model: TemperatureModel, theta0: FloatArray,
                       cache: dict[TemperatureModel, FloatArray]) -> FloatArray:
    """Evaluate each required model once on the figure's common grid"""
    values = cache.get(model)
    if values is None:
        values = evaluate_temperature(model, theta0)
        cache[model] = values
    return values


def relative_error(item: ErrorCurveConfig, theta0: FloatArray,
                   cache: dict[TemperatureModel, FloatArray]) -> FloatArray:
    """Compute absolute percentage error against a positive reference conversion"""
    reference = cached_temperature(item.reference_model, theta0, cache)
    approximation = cached_temperature(item.approximation_model, theta0, cache)
    return error_vector(100.0 * np.abs((approximation - reference) / reference))


def resolve_item(item: DisplayItem, theta0: FloatArray,
                 cache: dict[TemperatureModel, FloatArray]) -> ResolvedCurve:
    """Keep each displayed item associated with its resolved ordinate values"""
    if isinstance(item, CurveConfig):
        values = cached_temperature(item.temperature_model, theta0, cache)
    elif isinstance(item, ErrorCurveConfig):
        values = relative_error(item, theta0, cache)
    elif isinstance(item, ReferenceLineConfig):
        threshold = real_scalar(item.percent, "Error threshold")
        if threshold <= 0:
            raise ValueError("Error threshold must be positive")
        values = np.full_like(theta0, threshold)
    else:
        raise TypeError("Unsupported panel item")
    return ResolvedCurve(item, values)


def prepare_data(config: PlotSettings) -> PlotData:
    """Generate the common grid and retain each curve with its own settings"""
    validate_settings(config)
    theta0 = generate_theta0(config.grid)
    cache: dict[TemperatureModel, FloatArray] = {}
    panels = tuple(ResolvedPanel(panel, tuple(resolve_item(curve, theta0, cache) for curve in panel.curves))
                   for panel in config.panels)
    data = PlotData(theta0, panels)
    validate_data(data, config)
    return data


def draw_panel(ax: Axes, theta0: FloatArray, panel: ResolvedPanel, config: PlotSettings) -> None:
    """Draw resolved curves and legends in their configured tuple order"""
    pu.AxesConfigurator.apply(ax, panel.config.axes)
    plotter = pu.Plotter(ax)
    entries: list[pu.LegendEntry] = []
    reference_artists: list[tuple[Line2D, float]] = []
    for curve in panel.curves:
        artists = plotter.line(theta0, curve.theta, curve.config.style)
        if isinstance(curve.config, ReferenceLineConfig):
            reference_artists.append((artists[0], real_scalar(curve.config.percent, "Error threshold")))
        style = curve.config.style
        component = pu.ComponentConfig(
            type="line",
            style=pu.LineStyleConfig(colors=[style.color], linewidth=style.linewidth,
                                     linestyle=style.linestyle, alpha=style.alpha, zorder=style.zorder),
            show_legend=True,
            label_format=pu.QuantityLabel(prefix=curve.config.label, fmt=".4g", unit=""),
        )
        entries.append(pu.LegendEntry(component, artists[0]))
    # Extend reference lines across the final displayed range without changing limits
    limits = ax.get_xlim()
    for artist, threshold in reference_artists:
        artist.set_data(limits, (threshold, threshold))
    ax.set_xlim(limits)
    plotter.legend(entries, config.legend)


def draw_figure(data: PlotData, config: PlotSettings) -> Figure:
    """Draw both panels with shared x axes and close the figure on failure"""
    validate_data(data, config)
    layout = replace(config.layout, axes=layout_axes(config.panels))
    fig, axes = layout.create()
    try:
        for ax, panel in zip(axes.flat, data.panels):
            draw_panel(ax, data.theta0, panel, config)
        for ax in axes[:-1, 0]:
            ax.set_xlabel("")
            ax.tick_params(axis="x", which="both", labelbottom=False)
        return fig
    except Exception:
        plt.close(fig)
        raise


def write_pdf(data: PlotData, config: PlotSettings, output: Path) -> None:
    """Publish a complete PDF atomically and always close its figure"""
    if output.suffix.lower() != ".pdf":
        raise ValueError("Output must have a .pdf suffix")
    fig = draw_figure(data, config)
    temporary: Path | None = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=output.parent, prefix=".electron_temperature_", suffix=".pdf", delete=False) as file:
            temporary = Path(file.name)
        fig.savefig(temporary, format="pdf")
        temporary.replace(output)
    finally:
        plt.close(fig)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = build_plot_config()
        data = prepare_data(config)
        write_pdf(data, config, args.output)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    selectors = ", ".join(curve.config.label for panel in data.panels for curve in panel.curves)
    print(f"Saved: {args.output}; theta0=[{data.theta0[0]:g}, {data.theta0[-1]:g}]; "
          f"samples={data.theta0.size}; items={selectors}", flush=True)


if __name__ == "__main__":
    main()
