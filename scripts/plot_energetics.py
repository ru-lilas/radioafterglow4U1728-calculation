"""Plot energy and mass fractions against eps_B or eps_th using typed Python settings"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from module import plot_utils as pu
from module.utils import FileReader
from module.mydataclasses import QuantityData

SweepParameter = Literal["eps_b", "eps_th"]
FractionName = Literal["mass_fraction", "energy_fraction"]
FloatArray = NDArray[np.float64]
RTOL = 1e-10
REQUIRED = {"run_id", "model_index", "nu", "eps_b", "eps_th", "energy_fraction", "mass_fraction"}


@dataclass(frozen=True, slots=True)
class PanelConfig:
    quantity: FractionName
    axes: pu.AxesConfig


@dataclass(frozen=True, slots=True)
class TitleItem:
    field: str
    source: Literal["column", "attrs", "fixed"]
    display: pu.MetadataConfig
    validation: Literal["finite", "positive", "fraction"] = "finite"
    missing: Literal["required", "omit"] = "required"
    unit_key: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedTitleItem:
    item: TitleItem
    component: pu.MetadataComponent

    def text(self) -> str:
        """Format a validated quantity using the shared metadata component"""
        strings: list[str] = []
        self.component.add_strings(strings)
        return "".join(strings)


@dataclass(frozen=True, slots=True)
class PlotSettings:
    layout: pu.PlotLayout
    panels: tuple[PanelConfig, PanelConfig]
    point_styles: tuple[pu.ErrorbarStyle, ...]
    legend: pu.ComponentConfig
    frequency_label: pu.QuantityLabel
    reference: pu.LineStyle | None
    title_fontsize: float
    show_metadata: bool
    sweep_label: str
    fixed_title: TitleItem
    title_items: tuple[TitleItem, ...]


def build_plot_config(parameter: SweepParameter) -> PlotSettings:
    """Edit all figure, axis, marker and annotation settings here"""
    # Set either y scale to "linear" to include zero fraction values
    x = pu.AxisConfig(
        scale="log" if parameter == "eps_b" else "linear", limits=None,
        label=pu.LabelConfigure(r"$\varepsilon_{\mathrm{B}}$" if parameter == "eps_b"
                                else r"$\varepsilon_{\mathrm{th}}$", "", 20),
    )
    ticks = pu.TickStyle(labelsize=15, major_length=7, minor_length=4, pad=6)
    energy_panel = PanelConfig(
        quantity="energy_fraction",
        axes=pu.AxesConfig(x, pu.AxisConfig("log", None,
            pu.LabelConfigure(r"$E_{\mathrm{kin}}/E_{\mathrm{nuc}}$", "", 20)), ticks),
    )
    mass_panel = PanelConfig(
        quantity="mass_fraction",
        axes=pu.AxesConfig(x, pu.AxisConfig("log", None,
            pu.LabelConfigure(r"$M_{\mathrm{ej}}/M_{\mathrm{acc}}$", "", 20)), ticks),
    )
    # Change only this tuple to reverse the panel order
    panels = (
        mass_panel,
        energy_panel
    )
    colors = ("#005AFF", "#FF4B00", "#03AF7A", "#990099")
    styles = tuple(
        pu.ErrorbarStyle(marker=marker, markersize=6, markerfacecolor=color,
                         markeredgecolor=color, ecolor=color, markeredgewidth=1,
                         linestyle="none", alpha=0.85, zorder=3)
        for marker in ("o", "s", "^", "D") for color in colors
    )
    # Each item owns its source, label, format, units and validation policy
    title_items = (
        TitleItem("distance_value", "attrs", pu.MetadataConfig(True, "kpc",
            pu.QuantityLabel("D", ".4g", r"\mathrm{kpc}")), "positive", "omit", "distance_unit"),
        TitleItem("mu_e_mu", "column", pu.MetadataConfig(True, "",
            pu.QuantityLabel(r"\mu_e/\mu", ".4g", "")), "positive", "omit"),
        TitleItem("p_index", "column", pu.MetadataConfig(True, "",
            pu.QuantityLabel("p", ".1f", "")), "positive", "omit"),
        TitleItem("f_e", "column", pu.MetadataConfig(True, "",
            pu.QuantityLabel(r"f_e", ".1e", "")), "positive", "omit"),
        TitleItem("burnt_mass_fraction", "attrs", pu.MetadataConfig(True, "",
            pu.QuantityLabel(r"\mathrm{burnt\ mass\ fraction}", ".4g", "")), "fraction", "omit"),
    )
    fixed_title = TitleItem(
        "eps_th" if parameter == "eps_b" else "eps_b", "fixed",
        pu.MetadataConfig(True, "", pu.QuantityLabel(
            r"\varepsilon_{\mathrm{th}}" if parameter == "eps_b" else r"\varepsilon_{\mathrm{B}}",
            ".4g", "")), "fraction",
    )
    return PlotSettings(
        layout=pu.PlotLayout(
            figure=pu.FigureConfig((12, 5.5), dpi=150, layout_engine="constrained"),
            axes=panels[0].axes, subplots=pu.SubplotConfig(nrows=1, ncols=2, sharex=True, sharey=False),
        ),
        panels=panels, point_styles=styles,
        legend=pu.ComponentConfig(type="legend", style=pu.LegendStyle(fontsize=12, loc="best")),
        frequency_label=pu.QuantityLabel(prefix="", fmt=".4g", unit=r"\mathrm{GHz}", show_equals=False),
        reference=pu.LineStyle(color="#777777", linewidth=1, linestyle="--", zorder=1),
        title_fontsize=12, show_metadata=True,
        sweep_label=(r"$\varepsilon_{\mathrm{B}}$ sweep" if parameter == "eps_b"
                     else r"$\varepsilon_{\mathrm{th}}$ sweep"),
        fixed_title=fixed_title, title_items=title_items,
    )


def parse_parameter(value: str) -> SweepParameter:
    """Narrow a validated CLI selector to its concrete literal type"""
    if value == "eps_b":
        return "eps_b"
    if value == "eps_th":
        return "eps_th"
    raise ValueError(f"Unsupported sweep parameter: {value}")


def native_scalar(value: object) -> object:
    """Convert NumPy scalar wrappers without accepting arrays"""
    if isinstance(value, np.generic):
        native: object = value.item()
        if isinstance(native, np.generic):
            raise TypeError("Unsupported NumPy scalar precision")
        return native
    return value


def real_scalar(value: object) -> float:
    """Validate a finite real scalar before numeric conversion"""
    value = native_scalar(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Expected a real numeric value, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Expected a finite number")
    return number


def integer_identifier(value: object, *, minimum: int) -> int:
    """Reject fractional, boolean and invalid identifiers without truncation"""
    value = native_scalar(value)
    if isinstance(value, int) and not isinstance(value, bool):
        result = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        result = int(value)
    else:
        raise TypeError("Expected an integer identifier")
    if result < minimum:
        raise ValueError(f"Identifier must be at least {minimum}")
    return result


@dataclass(frozen=True, slots=True)
class PlotRow:
    run_id: int
    model_index: int
    nu: float
    eps_b: float
    eps_th: float
    energy_fraction: float
    mass_fraction: float

    @property
    def identity(self) -> str:
        """Describe a row using its scientific result identity"""
        return f"run_id={self.run_id}, nu={self.nu:g} GHz, model_index={self.model_index}"

    def x_value(self, parameter: SweepParameter) -> float:
        """Return the selected energy fraction"""
        return self.eps_b if parameter == "eps_b" else self.eps_th


@dataclass(frozen=True, slots=True)
class FrequencyGroup:
    nu: float
    rows: tuple[PlotRow, ...]
    x: FloatArray
    energy: FloatArray
    mass: FloatArray


    def fraction_values(self, quantity: FractionName) -> FloatArray:
        """Select the same fraction array for drawing and validation"""
        if quantity == "energy_fraction":
            return self.energy
        if quantity == "mass_fraction":
            return self.mass
        raise ValueError(f"Unsupported fraction: {quantity}")


@dataclass(frozen=True, slots=True)
class PlotData:
    parameter: SweepParameter
    fixed_parameter: SweepParameter
    fixed_value: float
    groups: tuple[FrequencyGroup, ...]
    metadata: dict[str, object]
    row_count: int
    title_columns: dict[str, tuple[object, ...]]
    title_rows: tuple[PlotRow, ...]


def metadata_flag(value: object) -> bool:
    """Interpret explicit provenance flags from CSV metadata"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise ValueError("Invalid provenance_complete metadata flag")


def prepare_data(df: pd.DataFrame, parameter: SweepParameter) -> PlotData:
    """Validate external data and prepare concrete arrays grouped by frequency"""
    if not df.columns.is_unique:
        raise ValueError("Duplicate column labels are not allowed")
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("The energetics table is empty")
    metadata: dict[str, object] = {}
    for key, value in df.attrs.items():
        if not isinstance(key, str):
            raise TypeError("Metadata keys must be strings")
        metadata[key] = value
    nu_unit = metadata.get("nu_unit")
    if not isinstance(nu_unit, str) or not nu_unit:
        raise ValueError("A frequency unit is required in nu_unit metadata")
    frequency_factor = real_scalar(u.Unit(nu_unit).to(u.GHz))
    rows: list[PlotRow] = []
    identities: set[tuple[int, float, int]] = set()
    run_parameters: dict[int, tuple[float, float]] = {}
    for position, record in enumerate(df.to_dict(orient="records")):
        try:
            row = PlotRow(
                integer_identifier(record["run_id"], minimum=1),
                integer_identifier(record["model_index"], minimum=0),
                real_scalar(record["nu"]) * frequency_factor,
                real_scalar(record["eps_b"]), real_scalar(record["eps_th"]),
                real_scalar(record["energy_fraction"]), real_scalar(record["mass_fraction"]),
            )
            if not math.isfinite(row.nu) or row.nu <= 0:
                raise ValueError(f"nu must be positive and finite: {row.identity}")
            if not 0 < row.eps_b <= 1 or not 0 < row.eps_th <= 1:
                raise ValueError(f"eps_b and eps_th must be in (0, 1]: {row.identity}")
            if row.energy_fraction < 0 or row.mass_fraction < 0:
                raise ValueError(f"Fractions must be nonnegative: {row.identity}")
            identity = (row.run_id, row.nu, row.model_index)
            if identity in identities:
                raise ValueError(f"Duplicate result identity: {row.identity}")
            identities.add(identity)
            pair = (row.eps_b, row.eps_th)
            previous = run_parameters.get(row.run_id)
            if previous is not None and not all(math.isclose(a, b, rel_tol=RTOL, abs_tol=0) for a, b in zip(pair, previous)):
                raise ValueError(f"Input parameters differ within RUN: {row.identity}")
            run_parameters[row.run_id] = pair
            rows.append(row)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Input row {position}, run_id={record.get('run_id')}, model_index={record.get('model_index')}: {error}") from error
    fixed_parameter: SweepParameter = "eps_th" if parameter == "eps_b" else "eps_b"
    fixed = rows[0].x_value(fixed_parameter)
    for row in rows:
        if not math.isclose(row.x_value(fixed_parameter), fixed, rel_tol=RTOL, abs_tol=0):
            raise ValueError(f"Fixed {fixed_parameter} varies: {row.identity}")
    groups: list[FrequencyGroup] = []
    for nu in sorted({row.nu for row in rows}):
        selected = tuple(row for row in rows if row.nu == nu)
        groups.append(FrequencyGroup(
            nu=nu, rows=selected,
            x=np.asarray([row.x_value(parameter) for row in selected], dtype=np.float64),
            energy=np.asarray([row.energy_fraction for row in selected], dtype=np.float64),
            mass=np.asarray([row.mass_fraction for row in selected], dtype=np.float64),
        ))
    if len({row.x_value(parameter) for row in rows}) == 1:
        print("WARNING: only one horizontal value is available; a sweep trend cannot be assessed", flush=True)
    if "provenance_complete" in metadata and not metadata_flag(metadata["provenance_complete"]):
        print("WARNING: input provenance is incomplete; stale results cannot be excluded", flush=True)
    columns: dict[str, tuple[object, ...]] = {}
    for name in df.columns:
        if not isinstance(name, str):
            raise TypeError("Column labels must be strings")
        columns[name] = tuple(require_series(df, name).tolist())
    return PlotData(parameter, fixed_parameter, fixed, tuple(groups), metadata, len(rows), columns, tuple(rows))


def validate_panels(config: PlotSettings) -> None:
    """Validate panel identities, layout and shared horizontal-axis settings"""
    subplots = config.layout.subplots
    if (subplots.nrows, subplots.ncols) != (1, 2) or len(config.panels) != 2:
        raise ValueError("Configure exactly two panels in a one-row, two-column layout")
    if {panel.quantity for panel in config.panels} != {"energy_fraction", "mass_fraction"}:
        raise ValueError("Configure each fraction exactly once")
    if subplots.sharex not in (True, "all", "row") or subplots.sharey not in (False, "none"):
        raise ValueError("Panels must share x and retain independent y axes")
    first_x = pu.AxesConfig._single_axis(config.panels[0].axes.x)
    for panel in config.panels:
        x = pu.AxesConfig._single_axis(panel.axes.x)
        pu.AxesConfig._single_axis(panel.axes.y)
        if x != first_x:
            raise ValueError("Shared x-axis settings must agree across panels")


def validate_scales(data: PlotData, config: PlotSettings) -> None:
    """Validate each fraction against its own panel settings in any panel order"""
    validate_panels(config)
    if len(data.groups) > len(config.point_styles):
        raise ValueError("Add more point styles in build_plot_config for the available frequencies")
    for panel in config.panels:
        x_config = pu.AxesConfig._single_axis(panel.axes.x)
        y_config = pu.AxesConfig._single_axis(panel.axes.y)
        for group in data.groups:
            y_values = group.fraction_values(panel.quantity)
            for row, x, y in zip(group.rows, group.x, y_values):
                if (x_config.scale == "log" and x <= 0) or (y_config.scale == "log" and y <= 0):
                    raise ValueError(f"Nonpositive value on logarithmic axis ({panel.quantity}): {row.identity}; select a linear scale in build_plot_config")
                for value, axis, column in ((x, x_config, data.parameter), (y, y_config, panel.quantity)):
                    if axis.limits is not None and not axis.limits[0] <= value <= axis.limits[1]:
                        raise ValueError(f"Configured {column} limits would hide a point: {row.identity}")


def require_series(df: pd.DataFrame, name: str) -> pd.Series:
    """Narrow a single column selection before processing its elements"""
    column: object = df[name]
    if not isinstance(column, pd.Series):
        raise TypeError(f"Expected a single Series for {name}")
    return column


def validate_title_scalar(value: object, item: TitleItem) -> float:
    """Validate a title value using the rule owned by its item"""
    number = real_scalar(value)
    if item.validation == "positive" and number <= 0:
        raise ValueError(f"{item.field} must be positive")
    if item.validation == "fraction" and not 0 < number <= 1:
        raise ValueError(f"{item.field} must be in (0, 1]")
    if item.validation not in ("finite", "positive", "fraction"):
        raise ValueError(f"Unsupported validation for {item.field}")
    return number


def resolve_title_item(data: PlotData, item: TitleItem) -> ResolvedTitleItem | None:
    """Resolve one declared source and validate a shared physical quantity"""
    if not item.display.show:
        return None
    if item.missing not in ("required", "omit"):
        raise ValueError(f"Unsupported missing-value policy for {item.field}")
    unit = item.display.unit
    if item.source == "fixed":
        if item.field != data.fixed_parameter:
            raise ValueError("Fixed title item disagrees with the selected sweep")
        value = validate_title_scalar(data.fixed_value, item)
    elif item.source == "column":
        values = data.title_columns.get(item.field)
        if values is None:
            if item.missing == "omit":
                return None
            raise ValueError(f"Missing title column: {item.field}")
        checked: list[float] = []
        for raw, row in zip(values, data.title_rows):
            try:
                checked.append(validate_title_scalar(raw, item))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Title {item.field}, {row.identity}: {error}") from error
        if not checked or len(checked) != data.row_count:
            raise ValueError(f"Incomplete title column: {item.field}")
        value = checked[0]
        for current, row in zip(checked, data.title_rows):
            if not math.isclose(current, value, rel_tol=RTOL, abs_tol=0):
                raise ValueError(f"Title {item.field} varies: {row.identity}")
    elif item.source == "attrs":
        has_value = item.field in data.metadata
        has_unit = item.unit_key is not None and item.unit_key in data.metadata
        if not has_value:
            if has_unit or item.missing == "required":
                raise ValueError(f"Missing title metadata: {item.field}")
            return None
        value = validate_title_scalar(data.metadata[item.field], item)
    else:
        raise ValueError(f"Unsupported title source: {item.source}")
    if item.unit_key is not None:
        raw_unit = data.metadata.get(item.unit_key)
        if not isinstance(raw_unit, str) or not raw_unit:
            raise ValueError(f"Title {item.field} requires unit metadata {item.unit_key}")
        value = validate_title_scalar(value * real_scalar(u.Unit(raw_unit).to(unit)), item)
    component = pu.MetadataComponent(QuantityData(value, unit), item.display)
    return ResolvedTitleItem(item, component)


def title_text(data: PlotData, config: PlotSettings) -> str:
    """Join configured title components without quantity-specific branches"""
    items = (config.fixed_title,) + (config.title_items if config.show_metadata else ())
    parts = [config.sweep_label]
    for item in items:
        resolved = resolve_title_item(data, item)
        if resolved is not None:
            parts.append(resolved.text())
    return ", ".join(parts)


def draw_panel(ax: Axes, data: PlotData, config: PlotSettings, panel: PanelConfig) -> None:
    """Draw one fraction panel using the shared frequency styles"""
    pu.AxesConfigurator.apply(ax, panel.axes)
    plotter = pu.Plotter(ax)
    legends: list[pu.LegendEntry] = []
    for frequency_index, group in enumerate(data.groups):
        y = group.fraction_values(panel.quantity)
        style = config.point_styles[frequency_index]
        container = plotter.errorbar(x=group.x, y=y, yerr=np.zeros_like(y), style=style)
        artist = container.lines[0]
        if artist is None:
            raise RuntimeError("The scatter point artist was not created")
        component = pu.ComponentConfig(
            type="errorbar", style=pu.ErrorbarStyleConfig(colors=[style.markerfacecolor]),
            show_legend=True, label_format=config.frequency_label,
        )
        legends.append(pu.LegendEntry(component=component, artist=artist, value=group.nu))
    if config.reference is not None:
        limits = ax.get_xlim()
        plotter.line(x=np.asarray(limits, dtype=np.float64), y=np.ones(2), style=config.reference)
        ax.set_xlim(limits)
    plotter.legend(legends, config.legend)


def draw_figure(data: PlotData, config: PlotSettings) -> Figure:
    """Create the figure and close it if any rendering step fails"""
    validate_scales(data, config)
    title = title_text(data, config)
    # Derive initial layout axes from the ordered panels, never a fixed quantity
    layout = replace(config.layout, axes=config.panels[0].axes)
    fig, axes = layout.create()
    try:
        for ax, panel in zip(axes.flat, config.panels):
            draw_panel(ax, data, config, panel)
        fig.suptitle(title, fontsize=config.title_fontsize)
        return fig
    except Exception:
        plt.close(fig)
        raise


def write_pdf(data: PlotData, config: PlotSettings, output: Path, *, input_path: Path) -> None:
    """Save atomically and close the figure on both success and failure"""
    if output.resolve() == input_path.resolve():
        raise ValueError("Output must not overwrite the input table")
    if output.suffix.lower() != ".pdf":
        raise ValueError("Output must have a .pdf suffix")
    fig = draw_figure(data, config)
    temporary: Path | None = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=output.parent, prefix=".energetics_plot_", suffix=".pdf", delete=False) as file:
            temporary = Path(file.name)
        fig.savefig(temporary, format="pdf")
        temporary.replace(output)
    finally:
        plt.close(fig)
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--parameter", type=parse_parameter, choices=("eps_b", "eps_th"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        parameter = parse_parameter(args.parameter)
        df: pd.DataFrame = FileReader.table_from_csv(args.input)
        data = prepare_data(df, parameter)
        config = build_plot_config(parameter)
        write_pdf(data, config, args.output, input_path=args.input)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Saved: {args.output}; plotted rows: {data.row_count}")
    for group in data.groups:
        print(f"{group.nu:g} GHz: {len(group.rows)} points per panel")


if __name__ == "__main__":
    main()
