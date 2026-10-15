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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from module import plot_utils as pu
from module.models import DimensionlessElectronTemperature, ElectronTemperature, MolecularWeight

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
class TemperaturePageConfig:
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


def validate_settings(config: TemperaturePageConfig) -> None:
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


def validate_data(data: PlotData, config: TemperaturePageConfig) -> None:
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


def prepare_data(config: TemperaturePageConfig) -> PlotData:
    """Generate the common grid and retain each curve with its own settings"""
    validate_settings(config)
    theta0 = generate_theta0(config.grid)
    cache: dict[TemperatureModel, FloatArray] = {}
    panels = tuple(ResolvedPanel(panel, tuple(resolve_item(curve, theta0, cache) for curve in panel.curves))
                   for panel in config.panels)
    data = PlotData(theta0, panels)
    validate_data(data, config)
    return data


def draw_panel(ax: Axes, theta0: FloatArray, panel: ResolvedPanel, config: TemperaturePageConfig) -> None:
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


def draw_figure(data: PlotData, config: TemperaturePageConfig) -> Figure:
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


@dataclass(frozen=True, slots=True)
class ShockCurveConfig:
    eps_th: float
    mu_e_mu: float
    label: str
    style: pu.LineStyle


@dataclass(frozen=True, slots=True)
class FixedCondition:
    parameter: Literal["eps_th", "mu_e_mu"]
    value: float
    label: pu.QuantityLabel


@dataclass(frozen=True, slots=True)
class BetaPanelConfig:
    quantity: Literal["theta", "theta0"]
    curves: tuple[ShockCurveConfig, ...]
    axes: pu.AxesConfig
    fixed: FixedCondition


@dataclass(frozen=True, slots=True)
class BetaPageConfig:
    grid: GridConfig
    panels: tuple[BetaPanelConfig, BetaPanelConfig, BetaPanelConfig, BetaPanelConfig]
    layout: pu.PlotLayout
    legend: pu.ComponentConfig


@dataclass(frozen=True, slots=True)
class TemperaturePair:
    theta0: FloatArray
    theta: FloatArray

    def select(self, quantity: Literal["theta", "theta0"]) -> FloatArray:
        """Select the same ordinate for axis validation and drawing"""
        if quantity == "theta":
            return self.theta
        if quantity == "theta0":
            return self.theta0
        raise ValueError(f"Unsupported temperature ordinate: {quantity}")


@dataclass(frozen=True, slots=True)
class ResolvedShockCurve:
    config: ShockCurveConfig
    temperatures: TemperaturePair


@dataclass(frozen=True, slots=True)
class ResolvedBetaPanel:
    config: BetaPanelConfig
    curves: tuple[ResolvedShockCurve, ...]


@dataclass(frozen=True, slots=True)
class BetaPlotData:
    beta_sh: FloatArray
    panels: tuple[ResolvedBetaPanel, ...]


@dataclass(frozen=True, slots=True)
class PlotSettings:
    pages: tuple[TemperaturePageConfig, BetaPageConfig]


@dataclass(frozen=True, slots=True)
class DocumentData:
    pages: tuple[PlotData, BetaPlotData]


def beta_layout_axes(panels: tuple[BetaPanelConfig, BetaPanelConfig, BetaPanelConfig, BetaPanelConfig]) -> pu.AxesConfig:
    """Derive common x settings and one y configuration for each row"""
    return pu.AxesConfig(
        x=pu.AxesConfig._single_axis(panels[0].axes.x),
        y=tuple(pu.AxesConfig._single_axis(panel.axes.y) for panel in panels[::2]),
        ticks=panels[0].axes.ticks,
    )


def build_plot_config() -> PlotSettings:
    """Configure both pages and all physical curve instances in one location"""
    grid = GridConfig(lower=1e-1, upper=1e2, samples=512, sampling="log")
    gp98 = CurveConfig(
        temperature_model="gp98", label="GP98",
        style=pu.LineStyle(color="#000000", linewidth=2),
    )
    hot = CurveConfig(
        temperature_model="hot", label=r"Hot: $\Theta=\Theta_0/3$",
        style=pu.LineStyle(color="#000000", linewidth=1, linestyle="--"),
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
    temperature_page = TemperaturePageConfig(
        grid=grid, panels=panels,
        layout=pu.PlotLayout(
            figure=pu.FigureConfig((6, 8), dpi=150, layout_engine="constrained"),
            axes=layout_axes(panels), subplots=pu.SubplotConfig(nrows=2, ncols=1, sharex=True, sharey=False),
        ),
        legend=pu.ComponentConfig(type="legend", style=pu.LegendStyle(fontsize=12, loc="best")),
    )

    grid = GridConfig(lower=5.0e-2, upper=5.0e-1, samples=512, sampling="log")
    styles = tuple(pu.LineStyle(color="#000000", linewidth=1.8, linestyle=line)
                   for line in ("--", "-", ":"))
    eps_label = pu.QuantityLabel(r"\varepsilon_{\mathrm{th}}", ".2f", "")
    ratio_label = pu.QuantityLabel(r"\mu_e/\mu", ".1f", "")
    fixed_ratio = FixedCondition("mu_e_mu", 2.0, ratio_label)
    fixed_eps = FixedCondition("eps_th", 0.40, eps_label)
    eps_curves = tuple(ShockCurveConfig(value, fixed_ratio.value, eps_label.to_text(value), style)
                       for value, style in zip((0.20, 0.40, 0.60), styles))
    ratio_curves = tuple(ShockCurveConfig(fixed_eps.value, value, ratio_label.to_text(value), style)
                         for value, style in zip((1.0, 2.0, 4.0), styles))
    x = pu.AxisConfig("log", None, pu.LabelConfigure(r"$\beta_{\mathrm{sh}}$", "", 20))
    ticks = pu.TickStyle(labelsize=13, major_length=7, minor_length=4, pad=6)
    theta_axes = pu.AxesConfig(x, pu.AxisConfig("log", None, pu.LabelConfigure(r"$\Theta$", "", 20)), ticks)
    theta0_axes = replace(theta_axes, y=pu.AxisConfig("log", None, pu.LabelConfigure(r"$\Theta_0$", "", 20)))
    theta_eps_panel = BetaPanelConfig("theta", eps_curves, theta_axes, fixed_ratio)
    theta_ratio_panel = BetaPanelConfig("theta", ratio_curves, theta_axes, fixed_eps)
    theta0_eps_panel = BetaPanelConfig("theta0", eps_curves, theta0_axes, fixed_ratio)
    theta0_ratio_panel = BetaPanelConfig("theta0", ratio_curves, theta0_axes, fixed_eps)
    panels = (theta_eps_panel, theta_ratio_panel, theta0_eps_panel, theta0_ratio_panel)
    beta_page = BetaPageConfig(
        grid=grid, panels=panels,
        layout=pu.PlotLayout(
            figure=pu.FigureConfig((10, 8), dpi=150, layout_engine="constrained"),
            axes=beta_layout_axes(panels),
            subplots=pu.SubplotConfig(nrows=2, ncols=2, sharex=True, sharey="row"),
        ),
        legend=pu.ComponentConfig(type="legend", style=pu.LegendStyle(fontsize=11, loc="best")),
    )
    return PlotSettings(pages=(temperature_page, beta_page))


def validate_beta_settings(config: BetaPageConfig) -> None:
    """Validate physical columns, panel quantities and shared axes on page two"""
    subplots = config.layout.subplots
    if (subplots.nrows, subplots.ncols) != (2, 2) or len(config.panels) != 4:
        raise ValueError("The beta page requires a two-row, two-column layout")
    if subplots.sharex not in (True, "all") or subplots.sharey != "row":
        raise ValueError("The beta page must share all x axes and y axes by row")
    lower = real_scalar(config.grid.lower, "beta_sh lower bound")
    upper = real_scalar(config.grid.upper, "beta_sh upper bound")
    if not 0 < lower < upper < 1:
        raise ValueError("Require 0 < beta_sh lower bound < upper bound < 1")
    expected = (("theta", "mu_e_mu"), ("theta", "eps_th"),
                ("theta0", "mu_e_mu"), ("theta0", "eps_th"))
    if tuple((panel.quantity, panel.fixed.parameter) for panel in config.panels) != expected:
        raise ValueError("Beta panels must follow the specified quantity and fixed-parameter mapping")
    first_x = pu.AxesConfig._single_axis(config.panels[0].axes.x)
    for panel in config.panels:
        if pu.AxesConfig._single_axis(panel.axes.x) != first_x:
            raise ValueError("All beta-page x settings must agree")
        if not panel.curves:
            raise ValueError("Each beta panel requires physical curves")
        fixed = real_scalar(panel.fixed.value, "Fixed beta-page parameter")
        required = 2.0 if panel.fixed.parameter == "mu_e_mu" else 0.40
        if not math.isclose(fixed, required, rel_tol=1e-10, abs_tol=0):
            raise ValueError(f"Fixed {panel.fixed.parameter} must be {required:g}")
        pairs: set[tuple[float, float]] = set()
        for curve in panel.curves:
            eps = real_scalar(curve.eps_th, "eps_th")
            ratio = real_scalar(curve.mu_e_mu, "mu_e_mu")
            if not 0 < eps <= 1 or ratio <= 0:
                raise ValueError("Require eps_th in (0, 1] and positive mu_e_mu")
            current = ratio if panel.fixed.parameter == "mu_e_mu" else eps
            if not math.isclose(current, fixed, rel_tol=1e-10, abs_tol=0):
                raise ValueError(f"Curve disagrees with fixed {panel.fixed.parameter}")
            if (eps, ratio) in pairs:
                raise ValueError("Duplicate beta-page parameter pair")
            pairs.add((eps, ratio))
            if not isinstance(curve.label, str) or not curve.label.strip():
                raise ValueError("Each physical curve requires a label")
    for upper_panel, lower_panel in zip(config.panels[:2], config.panels[2:]):
        if upper_panel.curves != lower_panel.curves or upper_panel.fixed != lower_panel.fixed:
            raise ValueError("Both rows in a beta-page column must share the same curve settings")
    for left, right in (config.panels[:2], config.panels[2:]):
        if pu.AxesConfig._single_axis(left.axes.y) != pu.AxesConfig._single_axis(right.axes.y):
            raise ValueError("Shared beta-page y settings must agree within each row")


def prepare_beta_data(config: BetaPageConfig) -> BetaPlotData:
    """Compute each distinct physical curve once and reuse it across panels"""
    validate_beta_settings(config)
    beta_sh = generate_theta0(config.grid)
    cache: dict[tuple[float, float], TemperaturePair] = {}
    panels: list[ResolvedBetaPanel] = []
    for panel in config.panels:
        curves: list[ResolvedShockCurve] = []
        for curve in panel.curves:
            key = (curve.eps_th, curve.mu_e_mu)
            pair = cache.get(key)
            if pair is None:
                effmass = real_scalar(
                    MolecularWeight(mu_e=curve.mu_e_mu, mu=1.0).to_effective_mass_ratio(), "effmass_ep",
                )
                if effmass <= 0:
                    raise ValueError("effmass_ep must be positive")
                raw: object = ElectronTemperature.theta0(effmass_ep=effmass, eps_th=curve.eps_th, beta_sh=beta_sh)
                theta0 = positive_vector(raw, "beta-page theta0")
                if theta0.shape != beta_sh.shape:
                    raise ValueError("beta-page theta0 shape differs from beta_sh")
                theta = evaluate_temperature("gp98", theta0)
                pair = TemperaturePair(theta0, theta)
                cache[key] = pair
            curves.append(ResolvedShockCurve(curve, pair))
        panels.append(ResolvedBetaPanel(panel, tuple(curves)))
    data = BetaPlotData(beta_sh, tuple(panels))
    validate_beta_data(data, config)
    return data


def validate_beta_data(data: BetaPlotData, config: BetaPageConfig) -> None:
    """Check every curve against its own ordinate and shared-axis configuration"""
    validate_beta_settings(config)
    beta_sh = positive_vector(data.beta_sh, "beta_sh")
    if np.any(beta_sh >= 1) or np.any(np.diff(beta_sh) <= 0):
        raise ValueError("beta_sh must be strictly increasing and below one")
    if tuple(panel.config for panel in data.panels) != config.panels:
        raise ValueError("Resolved beta panels disagree with their configuration")
    for panel in data.panels:
        if tuple(curve.config for curve in panel.curves) != panel.config.curves:
            raise ValueError("Resolved physical curves disagree with their panel")
        validate_axis(beta_sh, pu.AxesConfig._single_axis(panel.config.axes.x), "beta_sh")
        for curve in panel.curves:
            for label, raw in (("Theta_0", curve.temperatures.theta0), ("Theta", curve.temperatures.theta)):
                values = positive_vector(raw, label)
                if values.shape != beta_sh.shape:
                    raise ValueError(f"{label} shape differs from beta_sh")
            validate_axis(curve.temperatures.select(panel.config.quantity),
                          pu.AxesConfig._single_axis(panel.config.axes.y), panel.config.quantity)


def draw_beta_figure(data: BetaPlotData, config: BetaPageConfig) -> Figure:
    """Draw the four beta panels with row-wise shared temperature axes"""
    validate_beta_data(data, config)
    fig, axes = replace(config.layout, axes=beta_layout_axes(config.panels)).create()
    try:
        for index, (ax, panel) in enumerate(zip(axes.flat, data.panels)):
            pu.AxesConfigurator.apply(ax, panel.config.axes)
            plotter = pu.Plotter(ax)
            entries: list[pu.LegendEntry] = []
            for curve in panel.curves:
                style = curve.config.style
                artists = plotter.line(data.beta_sh, curve.temperatures.select(panel.config.quantity), style)
                component = pu.ComponentConfig(
                    type="line",
                    style=pu.LineStyleConfig(colors=[style.color], linewidth=style.linewidth,
                                             linestyle=style.linestyle, alpha=style.alpha, zorder=style.zorder),
                    show_legend=True, label_format=pu.QuantityLabel(curve.config.label, ".4g", ""),
                )
                entries.append(pu.LegendEntry(component, artists[0]))
            if index < config.layout.subplots.ncols:
                fixed = panel.config.fixed
                ax.set_title(fixed.label.to_text(fixed.value), fontsize=13)
            plotter.legend(entries, config.legend)
        for ax in axes[:-1, :].flat:
            ax.set_xlabel("")
            ax.tick_params(axis="x", which="both", labelbottom=False)
        for ax in axes[:, 1:].flat:
            ax.set_ylabel("")
            ax.tick_params(axis="y", which="both", labelleft=False)
        return fig
    except Exception:
        plt.close(fig)
        raise


def prepare_document(config: PlotSettings) -> DocumentData:
    """Validate and prepare both pages before any PDF is published"""
    if len(config.pages) != 2:
        raise ValueError("Configure exactly two PDF pages")
    return DocumentData((prepare_data(config.pages[0]), prepare_beta_data(config.pages[1])))


def write_pdf(data: DocumentData, config: PlotSettings, output: Path) -> None:
    """Write both PDF pages atomically and close each figure on failure"""
    if output.suffix.lower() != ".pdf":
        raise ValueError("Output must have a .pdf suffix")
    if len(config.pages) != 2 or len(data.pages) != 2:
        raise ValueError("Exactly two PDF pages are required")
    validate_data(data.pages[0], config.pages[0])
    validate_beta_data(data.pages[1], config.pages[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(dir=output.parent, prefix=".electron_temperature_", suffix=".pdf", delete=False) as file:
            temporary = Path(file.name)
        with PdfPages(temporary) as pdf:
            for page_data, page_config in zip(data.pages, config.pages):
                if isinstance(page_data, PlotData) and isinstance(page_config, TemperaturePageConfig):
                    fig = draw_figure(page_data, page_config)
                elif isinstance(page_data, BetaPlotData) and isinstance(page_config, BetaPageConfig):
                    fig = draw_beta_figure(page_data, page_config)
                else:
                    raise TypeError("Page data and configuration types disagree")
                try:
                    pdf.savefig(fig)
                finally:
                    plt.close(fig)
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = build_plot_config()
        data = prepare_document(config)
        write_pdf(data, config, args.output)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Saved: {args.output}; pages=2; "
          f"theta0=[{data.pages[0].theta0[0]:g}, {data.pages[0].theta0[-1]:g}], samples={data.pages[0].theta0.size}; "
          f"beta_sh=[{data.pages[1].beta_sh[0]:g}, {data.pages[1].beta_sh[-1]:g}], samples={data.pages[1].beta_sh.size}",
          flush=True)


if __name__ == "__main__":
    main()
