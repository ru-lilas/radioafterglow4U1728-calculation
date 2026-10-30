from __future__ import annotations

import argparse
import csv
import math
from tempfile import NamedTemporaryFile
from typing import Literal

import astropy.units as u
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize, to_hex
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from dataclasses import dataclass, fields, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from module import calculation_lightcurve as clc
from module import plot_utils as pu
from module.models import ColumnNames, FluxName, MetadataNames, ModelParameters, SwitchKeys
from module.mydataclasses import LongformatDataFrame, QuantityArray1D, QuantityData
from module.observation import LongformatLightcurve, ObservedLightcurve
from module.utils import FileReader
from module.timing_utils import timed
from module import style_templates as st

try:
    from . import build_input
except ImportError:
    import build_input

style_continuous = pu.LineStyleConfig(
    colors=[
        "#FF4B00",
        "#005AFF"
    ],
    linestyle="-",
    linewidth=1.0
)
style_thermal = replace(
    style_continuous,
    linestyle=":",
)
style_binned = st.lc_binned
style_observed = pu.ErrorbarStyleConfig(
    colors=[
        "#FF4B00",
        "#005AFF"
    ],
    marker="o",
    markersize= 6.0,
    markeredgewidth= 1.0,
    elinewidth= 1.0,
    capsize= 1.0,
    capthick= 1.0,
    linestyle= "none",
    alpha= None,
    zorder= 0.0,
)
style_background = replace(
    style_continuous,
    colors=[
        "#000000"
    ],
)

@dataclass(frozen=True, slots=True)
class PlotMetadata:
    a_wind: QuantityData
    beta_sh: QuantityData
    eps_b: QuantityData
    eps_th: QuantityData
    mu_e_mu: QuantityData
    distance: QuantityData
    p_index: QuantityData
    f_e: QuantityData
    chi_squared: QuantityData | None

    @classmethod
    def from_model(
        cls,
        model: ModelParameters,
        chi_squared: float | None = None,
    ) -> PlotMetadata:
        return cls(
            a_wind=QuantityData.from_Quantity(model.a_wind.item()),
            beta_sh=QuantityData(model.beta_sh.item(), ""),
            eps_b=QuantityData(model.eps_b, ""),
            eps_th=QuantityData(model.eps_th, ""),
            mu_e_mu=QuantityData(model.molecular_weight.to_ratio(), ""),
            distance=model.distance,
            p_index=QuantityData(model.powerlaw.p_index, ""),
            f_e=QuantityData(model.powerlaw.f_e, ""),
            chi_squared=(None if chi_squared is None else QuantityData(chi_squared, "")),
        )

    def to_dict(self) -> dict[str, QuantityData]:
        values: dict[str, QuantityData] = {}
        for field in fields(self):
            quantity = getattr(self, field.name)
            if quantity is not None:
                values[field.name] = quantity
        return values

@dataclass(frozen=True, slots=True)
class SweepSettings:
    components: tuple[FluxName, ...] = (FluxName.FNU_TOTAL,)
    show_binned: bool = False
    normalization: Literal["log", "linear"] = "log"
    cmap: str = "cividis"
    component_styles: tuple[tuple[FluxName, pu.LineStyle], ...] = (
        (FluxName.FNU_TOTAL, replace(style_continuous.resolve(0), linestyle="-")),
        (FluxName.FNU_THERMAL, style_thermal.resolve(0)),
        (FluxName.FNU_POWERLAW, style_continuous.resolve(0)),
    )


def build_sweep_settings() -> SweepSettings:
    """Configure sweep curves and color normalization directly in Python"""
    return SweepSettings(normalization="linear")


SweepParameter = Literal["f_e", "p_index"]

SWEEP_LABELS: dict[SweepParameter, pu.QuantityLabel] = {
    "f_e": pu.QuantityLabel(r"f_{\mathrm{e}}", ".1e", ""),
    "p_index": pu.QuantityLabel("p", ".2f", ""),
}


def validate_sweep_parameter(value: object) -> SweepParameter:
    """Narrow the plot selector independently of the generator's other axes"""
    if isinstance(value, str):
        if value == "f_e":
            return "f_e"
        if value == "p_index":
            return "p_index"
    raise ValueError(f"Unsupported lightcurve sweep parameter: {value!r}")


def model_sweep_value(model: ModelParameters, parameter: SweepParameter) -> float:
    """Read one explicitly selected model parameter as a finite real scalar"""
    parameter = validate_sweep_parameter(parameter)
    if parameter == "f_e":
        return real_scalar(model.powerlaw.f_e)
    return real_scalar(model.powerlaw.p_index)


def sweep_title_fields(parameter: SweepParameter) -> tuple[str, ...]:
    """Exclude the varied field from candidate shared title quantities"""
    parameter = validate_sweep_parameter(parameter)
    fixed = "p_index" if parameter == "f_e" else "f_e"
    return ("eps_b", "eps_th", "mu_e_mu", "distance", fixed)


def validate_powerlaw_metadata(df: pd.DataFrame, path: Path) -> None:
    """Require valid power-law attrs before the model reader converts scalars"""
    for parameter, key in (("f_e", MetadataNames.F_E), ("p_index", MetadataNames.P_INDEX)):
        selector = validate_sweep_parameter(parameter)
        if key not in df.attrs:
            raise ValueError(f"Missing parameter-table metadata {parameter}: {path}")
        try:
            value = real_scalar(df.attrs[key])
            build_input.validate_sweep_values([value], parameter=selector)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid parameter-table metadata {parameter}: {path}: {error}") from error


@dataclass(frozen=True, slots=True)
class SweepSource:
    """Describe one sweep manifest and its RUN-relative result locations"""

    parameter: SweepParameter
    sweep_index: Path
    minimum_chi_squared_path: Path
    model_parameter_table_path: Path
    candidate_lightcurves_path: Path
    binned_lightcurves_path: Path | None = None


@dataclass(frozen=True, slots=True)
class SweepPanelSpec:
    """Keep source selection and presentation attached to one panel position"""

    row: int
    col: int
    source: SweepSource
    frequency: QuantityData
    axes: pu.AxesConfig
    settings: SweepSettings


def normalized_result_path(path: Path) -> Path:
    """Normalize a RUN-relative file path lexically without reading files"""
    if not isinstance(path, Path):
        raise TypeError("Result paths must be Path instances")
    if path.is_absolute():
        raise ValueError(f"Result path must be RUN-relative: {path}")
    parts: list[str] = []
    for part in path.parts:
        if part == "..":
            if not parts:
                raise ValueError(f"Result path escapes its RUN: {path}")
            parts.pop()
        elif part != ".":
            parts.append(part)
    if not parts:
        raise ValueError(f"Result path must name a file: {path}")
    return Path(*parts)


def validate_sweep_source(source: SweepSource) -> None:
    """Check source configuration without opening the manifest or results"""
    if not isinstance(source, SweepSource):
        raise TypeError("Expected a SweepSource instance")
    if source.parameter not in ("f_e", "p_index"):
        raise ValueError(f"Unsupported sweep parameter: {source.parameter!r}")
    if not isinstance(source.sweep_index, Path) or not source.sweep_index.name:
        raise ValueError("The sweep manifest must have an explicit file path")
    for path in (source.minimum_chi_squared_path, source.model_parameter_table_path,
                 source.candidate_lightcurves_path, source.binned_lightcurves_path):
        if path is not None:
            normalized_result_path(path)
    for path in (source.minimum_chi_squared_path, source.model_parameter_table_path,
                 source.candidate_lightcurves_path):
        if path is None:
            raise ValueError("All three required result paths must be supplied")


def validate_sweep_panels(panels: tuple[SweepPanelSpec, ...]) -> tuple[int, int]:
    """Validate panel configuration and return its rectangular grid dimensions"""
    if not isinstance(panels, tuple) or not panels:
        raise ValueError("Configure a nonempty tuple of sweep panels")
    positions: set[tuple[int, int]] = set()
    source_signatures: dict[Path, tuple[SweepParameter, tuple[Path | None, ...]]] = {}
    supported = (FluxName.FNU_TOTAL, FluxName.FNU_THERMAL, FluxName.FNU_POWERLAW)
    for panel in panels:
        if not isinstance(panel, SweepPanelSpec):
            raise TypeError("Expected SweepPanelSpec instances")
        for name, index in (("row", panel.row), ("col", panel.col)):
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError(f"Panel {name} must be a nonnegative integer: {index!r}")
        position = (panel.row, panel.col)
        if position in positions:
            raise ValueError(f"Duplicate panel position: {position}")
        positions.add(position)
        validate_sweep_source(panel.source)
        source = panel.source
        # Resolve the manifest only; result paths are still relative to an unknown RUN
        manifest = source.sweep_index.resolve()
        result_paths = tuple(
            None if path is None else normalized_result_path(path)
            for path in (source.minimum_chi_squared_path, source.model_parameter_table_path,
                         source.candidate_lightcurves_path, source.binned_lightcurves_path)
        )
        signature = (source.parameter, result_paths)
        previous = source_signatures.get(manifest)
        if previous is not None and previous != signature:
            raise ValueError(f"Conflicting source specifications for manifest: {manifest}")
        source_signatures[manifest] = signature
        if not isinstance(panel.frequency, QuantityData):
            raise TypeError(f"Panel {position}: frequency must be QuantityData")
        frequency = panel.frequency
        if not isinstance(frequency.unit, str) or not frequency.unit:
            raise ValueError(f"Panel {position}: explicit frequency unit is required")
        value = real_scalar(frequency.value)
        factor = real_scalar(u.Unit(frequency.unit).to(u.GHz))
        if real_scalar(value * factor) <= 0:
            raise ValueError(f"Panel {position}: frequency must be positive")
        if (not isinstance(panel.axes, pu.AxesConfig)
                or not isinstance(panel.axes.x, pu.AxisConfig)
                or not isinstance(panel.axes.y, pu.AxisConfig)):
            raise ValueError(f"Panel {position}: use concrete single-panel axis configurations")
        settings = panel.settings
        if not isinstance(settings, SweepSettings):
            raise TypeError(f"Panel {position}: expected SweepSettings")
        if (not settings.components or len(set(settings.components)) != len(settings.components)
                or any(component not in supported for component in settings.components)):
            raise ValueError(f"Panel {position}: select distinct supported continuous components")
        styles = dict(settings.component_styles)
        if (len(styles) != len(settings.component_styles)
                or any(component not in styles for component in settings.components)
                or any(not isinstance(styles[component], pu.LineStyle)
                       for component in settings.components)):
            raise ValueError(f"Panel {position}: configure one resolved style per selected component")
        if not isinstance(settings.show_binned, bool):
            raise TypeError(f"Panel {position}: show_binned must be boolean")
        if settings.show_binned and source.binned_lightcurves_path is None:
            raise ValueError(f"Panel {position}: binned drawing requires a source binned path")
        if settings.cmap not in colormaps:
            raise ValueError(f"Panel {position}: unknown colormap {settings.cmap!r}")
    nrows = max(row for row, _ in positions) + 1
    ncols = max(col for _, col in positions) + 1
    if len(positions) != nrows * ncols:
        raise ValueError("Panel positions must fill a rectangular grid without gaps")
    return nrows, ncols


def build_sweep_panels(
    *,
    rows: tuple[tuple[SweepSource, SweepSettings], ...],
    frequencies: tuple[QuantityData, ...],
    axes: pu.AxesConfig,
) -> tuple[SweepPanelSpec, ...]:
    """Build any number of source rows with explicit frequency columns

    Individual returned instances may be moved or restyled with replace before
    passing the tuple to draw_sweep. Use SweepPanelSpec directly for layouts
    with different frequencies in different rows
    """
    if not rows or not frequencies:
        raise ValueError("Supply source rows and explicit frequencies")
    x = axes.x
    y = axes.y
    if isinstance(x, tuple):
        if len(x) == 1:
            x = x[0]
        elif len(x) != len(frequencies):
            raise ValueError("Column axes must match the configured frequencies")
    if isinstance(y, tuple):
        if len(y) == 1:
            y = y[0]
        elif len(y) != len(rows):
            raise ValueError("Row axes must match the configured sources")
    concrete_axes = replace(axes, x=x, y=y)
    panels = tuple(
        SweepPanelSpec(row, col, source, frequency,
                       concrete_axes.for_subplot(row, col), settings)
        for row, (source, settings) in enumerate(rows)
        for col, frequency in enumerate(frequencies)
    )
    validate_sweep_panels(panels)
    return panels

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("bestfit", "candidates", "sweep"), default="bestfit")
    for name in ("input", "candidate_lightcurves", "model_parameter_table",
                 "minimum_chi_squared", "binned_lightcurves"):
        aliases = [f"--{name}"]
        if name == "binned_lightcurves":
            aliases.append("--candidate_binned_lightcurves")
        parser.add_argument(*aliases, dest=name, type=Path)
    for name in ("obslc", "plotconfig", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--parameter", choices=("f_e", "p_index"))
    parser.add_argument("--sweep-index", type=Path)
    parser.add_argument("--frequencies", type=float, nargs="+", metavar="GHz")
    for name in ("minimum_chi_squared", "model_parameter_table", "candidate_lightcurves",
                 "binned_lightcurves"):
        parser.add_argument("--" + name.replace("_", "-") + "-path", type=Path)
    args = parser.parse_args()
    single = ("input", "candidate_lightcurves", "model_parameter_table",
              "minimum_chi_squared", "binned_lightcurves")
    sweep = ("sweep_index", "parameter", "frequencies", "minimum_chi_squared_path",
             "model_parameter_table_path", "candidate_lightcurves_path", "binned_lightcurves_path")
    if args.mode == "sweep":
        for name in single:
            if getattr(args, name) is not None:
                parser.error(f"--mode sweep cannot use --{name}")
        for name in ("parameter", "sweep_index", "minimum_chi_squared_path",
                     "model_parameter_table_path", "candidate_lightcurves_path"):
            if getattr(args, name) is None:
                parser.error(f"--mode sweep requires --{name.replace('_', '-')}")
    else:
        for name in sweep:
            if getattr(args, name) is not None:
                parser.error(f"--{name.replace('_', '-')} requires --mode sweep")
        for name in single[1:]:
            if getattr(args, name) is None:
                parser.error(f"--mode {args.mode} requires --{name}")
    return args


def select_model_df(
    df: pd.DataFrame,
    model_index: int,
    nu_value: float,
) -> pd.DataFrame:
    selected = df.loc[df[ColumnNames.MODEL_INDEX] == model_index].copy()
    if selected.empty:
        raise ValueError(f"Missing model_index={model_index}, nu={nu_value}")
    selected.attrs = dict(df.attrs)
    selected.attrs.update({
        ColumnNames.NU: nu_value,
        ColumnNames.MODEL_INDEX: model_index,
    })
    return selected


def build_continuous_lightcurve(
    df: pd.DataFrame,
    model_index: int,
    nu: QuantityData,
    flux_name: FluxName,
) -> clc.ContinuousLightcurve:
    return clc.ContinuousLightcurve(
        model_index=model_index,
        nu=nu,
        t=QuantityArray1D.from_df(df, ColumnNames.T, unit_key=MetadataNames.T_UNIT),
        fnu_net=QuantityArray1D.from_df(df, flux_name, unit_key=MetadataNames.FNU_UNIT),
    )


def plot_observed(
    plotter: pu.Plotter, observed: ObservedLightcurve, col: int,
    *, x_unit: str, y_unit: str, sweep: bool = False,
) -> None:
    """Draw shared observations and background once per panel"""
    plotter.line(x=observed.t.to_value_in(x_unit),
                 y=observed.fnu_background.to_value_in(y_unit),
                 style=style_background.resolve(0))
    style = style_observed.resolve(col % len(style_observed.colors))
    if sweep:
        style = replace(style, markerfacecolor="#222222", markeredgecolor="#222222",
                        ecolor="#222222", zorder=5)
    plotter.errorbar(x=observed.t.to_value_in(x_unit),
                     y=observed.fnu_total.to_value_in(y_unit),
                     yerr=observed.fnu_total_err.to_value_in(y_unit), style=style)


def plot_model_curve(
    plotter: pu.Plotter, continuous: clc.ContinuousLightcurve,
    observed: ObservedLightcurve, style: pu.LineStyle,
    *, x_unit: str, y_unit: str,
) -> None:
    """Draw one resolved net curve with its observed background added once"""
    plotter.line(x=continuous.t.to_value_in(x_unit),
                 y=continuous.fnu_with_bg(observed.fnu_background.index(0)).to_value_in(y_unit),
                 style=style)


def plot_binned_curve(
    plotter: pu.Plotter, binned: clc.BinnedLightcurveWithBackground,
    observed: ObservedLightcurve, style: pu.LineStyle,
    *, x_unit: str, y_unit: str,
) -> None:
    """Draw one background-inclusive binned curve"""
    plotter.step(edges=binned.bin_edges.to_value_in(x_unit),
                 values=binned.fnu_with_bg.to_value_in(y_unit), style=style,
                 baseline=observed.fnu_background.to_value_in(y_unit)[0])


def add_model_metadata(
    plotter: pu.Plotter,
    metadata: PlotMetadata,
    plotconf: pu.PlotConfig,
    synchrotron_label: str,
) -> None:
    strings: list[str] = []
    for name, quantity in metadata.to_dict().items():
        pu.MetadataComponent(
            quantity=quantity,
            config=plotconf.metadata.items[name],
        ).add_strings(strings)
    plotter.title(strings=strings, fontsize=plotconf.metadata.fontsize)


RTOL = 1e-10
CurveKey = tuple[int, float, int]


@dataclass(frozen=True, slots=True)
class SweepRun:
    run_id: int
    run_name: str
    input_path: Path
    config_hash: str
    sweep_value: float


def real_scalar(value: object) -> float:
    """Narrow finite real scalars without accepting boolean or string coercion"""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Expected real scalar, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Expected finite scalar, got {number}")
    return number


def integer_identifier(value: object, *, minimum: int = 0) -> int:
    """Validate an integer identifier without truncation"""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, int) and not isinstance(value, bool):
        if value < minimum:
            raise ValueError(f"Identifier must be at least {minimum}")
        return value
    number = real_scalar(value)
    if not number.is_integer() or number < minimum:
        raise ValueError(f"Invalid integer identifier: {value}")
    return int(number)


def require_series(df: pd.DataFrame, name: str) -> pd.Series:
    """Select exactly one named column"""
    column: object = df[name]
    if not isinstance(column, pd.Series):
        raise TypeError(f"Expected one Series for {name}")
    return column


def numeric_values(df: pd.DataFrame, name: str) -> NDArray[np.float64]:
    """Validate a numeric column before array construction"""
    return np.asarray([real_scalar(value) for value in require_series(df, name)], dtype=np.float64)


def read_table(path: Path) -> pd.DataFrame:
    """Check original headers before pandas can rename duplicate labels"""
    with path.open(encoding="utf-8", newline="") as file:
        header = next(csv.reader(line for line in file if not line.startswith("#")), [])
    if not header or len(set(header)) != len(header):
        raise ValueError(f"Missing or duplicate column labels: {path}")
    df = FileReader.table_from_csv(path)
    # The generic metadata reader parses numeric-looking strings as floats
    # Preserve fingerprint bytes rather than coercing them through numeric types
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.startswith("#"):
                break
            key, separator, value = line[1:].strip().partition("=")
            if separator and key.strip() == "config_hash":
                df.attrs["config_hash"] = value.strip()
    if df.empty or not df.columns.is_unique:
        raise ValueError(f"Empty or ambiguous table: {path}")
    return df


def unit_from(df: pd.DataFrame, key: str, target: str) -> float:
    """Require explicit dimensionally compatible unit metadata"""
    value: object = df.attrs.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing unit metadata: {key}")
    return real_scalar(u.Unit(value).to(u.Unit(target)))


def frequency_groups(df: pd.DataFrame) -> dict[float, pd.DataFrame]:
    """Normalize frequencies to GHz and explicitly preserve attrs"""
    values = numeric_values(df, ColumnNames.NU) * unit_from(df, MetadataNames.NU_UNIT, "GHz")
    if np.any(values <= 0):
        raise ValueError("Frequencies must be positive")
    work = df.copy()
    work[ColumnNames.NU] = values
    work.attrs = dict(df.attrs)
    work.attrs[MetadataNames.NU_UNIT] = "GHz"
    groups: dict[float, pd.DataFrame] = {}
    for frequency in sorted(set(values.tolist())):
        nu = real_scalar(frequency)
        selected = work.loc[work[ColumnNames.NU] == nu].copy()
        selected.attrs = dict(work.attrs)
        groups[nu] = selected
    return groups


def match_frequency(available: list[float], requested: float) -> float:
    """Resolve one positive GHz frequency without ambiguous matching"""
    nu = real_scalar(requested)
    if nu <= 0:
        raise ValueError("Requested frequency must be positive")
    matches = [value for value in available if math.isclose(value, nu, rel_tol=RTOL, abs_tol=0)]
    if len(matches) != 1:
        raise ValueError(f"Missing or ambiguous frequency {nu:g} GHz; available: {available}")
    return matches[0]


def select_frequencies(available: list[float], requested: list[float] | None) -> list[float]:
    """Return deterministic ascending unique frequency selections"""
    if requested is None:
        return sorted(available)
    if not requested:
        raise ValueError("At least one frequency is required")
    return sorted({match_frequency(available, nu) for nu in requested})


def run_result_path(run: SweepRun, relative: Path) -> Path:
    """Resolve a result path without permitting escape from its RUN"""
    root = run.input_path.resolve().parent
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Result path must remain inside {root}: {relative}")
    return path


def read_sweep_runs(path: Path, *, parameter: SweepParameter) -> list[SweepRun]:
    """Validate one selected manifest and its full input family before results"""
    parameter = validate_sweep_parameter(parameter)
    spec = build_input.SWEEP_SPECS[parameter]
    expected = spec.manifest_fields
    runs: list[SweepRun] = []
    families: list[object] = []
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != expected:
            raise ValueError(f"Expected {parameter} manifest columns {expected}: {path}")
        for line_number, row in enumerate(reader, start=2):
            try:
                if None in row or any(value is None or value == "" for value in row.values()):
                    raise ValueError(f"Malformed manifest record: {row}")
                # Manifest cells are textual; int rejects fractional identifier spelling
                run_id = integer_identifier(int(row["run_id"]), minimum=1)
                value = build_input.validate_sweep_values(
                    [float(row[parameter])], parameter=parameter,
                )[0]
                run = SweepRun(run_id, row["run_name"], Path(row["input_path"]),
                               row["config_hash"], value)
                if run.run_name != f"run_{run_id:06d}" or run.input_path.parent.name != run.run_name:
                    raise ValueError(f"Conflicting RUN identity: {row}")
                if row["status"] not in ("created", "reused"):
                    raise ValueError(f"Invalid status for {run.run_name}")
                if any(old.run_id == run_id or old.input_path.resolve() == run.input_path.resolve()
                       or math.isclose(old.sweep_value, value, rel_tol=RTOL, abs_tol=0) for old in runs):
                    raise ValueError(f"Duplicate or ambiguous sweep assignment: {row}")
                config = build_input.load_yaml(run.input_path)
                # The generator intentionally shares this power-law validator for both axes
                build_input.validate_f_e_input(config)
                actual = real_scalar(build_input.get_nested_value(config, spec.yaml_path))
                if not math.isclose(actual, value, rel_tol=RTOL, abs_tol=0):
                    raise ValueError(f"Manifest {parameter} disagrees with {run.input_path}")
                if build_input.calculate_config_hash(config) != run.config_hash:
                    raise ValueError(f"Manifest hash disagrees with {run.input_path}")
                family = build_input.sweep_family(config, parameter=parameter)
                if families and family != families[0]:
                    raise ValueError(f"Fixed input settings differ: {run.input_path}")
                families.append(family)
                runs.append(run)
            except (KeyError, TypeError, ValueError, OSError) as error:
                raise ValueError(
                    f"{parameter} manifest {path}, record {line_number}, "
                    f"RUN={row.get('run_name')}, input={row.get('input_path')}: {error}"
                ) from error
    if not runs:
        raise ValueError(f"Empty {parameter} sweep manifest: {path}")
    return sorted(runs, key=lambda run: run.sweep_value)


def validate_provenance(df: pd.DataFrame, run: SweepRun, path: Path) -> None:
    """Reject available result fingerprint mismatches and disclose missing provenance"""
    hashes: list[object] = []
    if "config_hash" in df.attrs:
        hashes.append(df.attrs["config_hash"])
    if "config_hash" in df.columns:
        hashes.extend(require_series(df, "config_hash").tolist())
    if not hashes:
        print(f"WARNING: {run.run_name}: result freshness unverified: {path}", flush=True)
    for value in hashes:
        if not isinstance(value, str) or value != run.config_hash:
            raise ValueError(f"Result config_hash mismatch: {path}")
    config = build_input.load_yaml(run.input_path)
    for name in ("f_e", "eps_b", "eps_th", "p_index", "mu", "mu_e"):
        candidates: list[object] = []
        if name in df.attrs:
            candidates.append(df.attrs[name])
        if name in df.columns:
            candidates.extend(require_series(df, name).tolist())
        if candidates:
            if name in ("f_e", "p_index"):
                key = build_input.SWEEP_SPECS[validate_sweep_parameter(name)].yaml_path
            elif name in ("mu", "mu_e"):
                key = f"model_parameters.shock.common.molecular_weight.{name}"
            else:
                key = f"model_parameters.shock.common.{name}"
            expected = real_scalar(build_input.get_nested_value(config, key))
            if any(not math.isclose(real_scalar(value), expected, rel_tol=RTOL, abs_tol=0)
                   for value in candidates):
                raise ValueError(f"Result {name} disagrees with input: {path}")


def validate_samples(df: pd.DataFrame, *, binned: bool = False) -> None:
    """Validate identities and ordered samples within every source model"""
    unit_from(df, MetadataNames.T_UNIT, "min")
    unit_from(df, MetadataNames.FNU_UNIT, "mJy")
    for name in df.columns:
        if isinstance(name, str) and name.startswith("fnu_"):
            numeric_values(df, name)
    indices = [integer_identifier(value) for value in require_series(df, ColumnNames.MODEL_INDEX)]
    time_name = ColumnNames.T_LEFT if binned else ColumnNames.T
    times = numeric_values(df, time_name)
    for index in set(indices):
        mask = np.asarray([value == index for value in indices], dtype=np.bool_)
        t = times[mask]
        if t.size == 0 or np.any(np.diff(t) <= 0):
            raise ValueError(f"Duplicate or unordered samples for model_index={index}")
        if binned:
            right = numeric_values(df, ColumnNames.T_RIGHT)[mask]
            center = numeric_values(df, ColumnNames.T_CENTER)[mask]
            if (np.any(right <= t) or np.any(center < t) or np.any(center > right)
                    or not np.allclose(right[:-1], t[1:], rtol=RTOL, atol=0)):
                raise ValueError(f"Invalid or noncontiguous bins for model_index={index}")


def validate_observed(observed: ObservedLightcurve) -> None:
    """Check observation arrays and the constant-background curve convention"""
    t = np.asarray(observed.t.to_value_in("min"), dtype=float)
    arrays = [np.asarray(quantity.to_value_in("mJy"), dtype=float) for quantity in
              (observed.fnu_total, observed.fnu_total_err, observed.fnu_background)]
    if (t.size == 0 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0)
            or any(values.shape != t.shape or not np.all(np.isfinite(values)) for values in arrays)
            or np.any(arrays[1] < 0)):
        raise ValueError("Invalid observation times, flux or uncertainties")
    if not np.allclose(arrays[2], arrays[2][0], rtol=RTOL, atol=0):
        raise ValueError("Model background addition requires a constant observed background")


def color_mapping(
    runs: list[SweepRun],
    settings: SweepSettings,
) -> dict[int, str]:
    """Assign evenly spaced colormap colors in ascending sweep-value order"""
    ordered = sorted(runs, key=lambda run: run.sweep_value)
    cmap = colormaps[settings.cmap]
    positions = (
        np.linspace(0.0, 1.0, len(ordered))
        if len(ordered) > 1
        else np.array([0.5])
    )

    return {
        run.run_id: to_hex(cmap(float(position)))
        for run, position in zip(ordered, positions)
    }

def check_axes(ax_config: pu.AxesConfig, x: object, y: object) -> None:
    """Reject invalid displayed values instead of silently flooring them"""
    for values, config in ((x, pu.AxesConfig._single_axis(ax_config.x)),
                           (y, pu.AxesConfig._single_axis(ax_config.y))):
        array = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(array)) or (config.scale == "log" and np.any(array <= 0)):
            raise ValueError("Nonfinite or nonpositive displayed value on configured axes")


def save_figure(fig: Figure, output: Path, sources: set[Path]) -> None:
    """Publish one PDF atomically while protecting inputs and existing output"""
    temporary: Path | None = None
    try:
        if output.suffix.lower() != ".pdf" or output.resolve() in {path.resolve() for path in sources}:
            raise ValueError("Output must be a PDF distinct from every source file")
        output.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=output.parent, prefix=".lightcurves_", suffix=".pdf", delete=False) as file:
            temporary = Path(file.name)
        fig.savefig(temporary, format="pdf")
        temporary.replace(output)
    finally:
        plt.close(fig)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def draw_bestfit(args: argparse.Namespace, plotconf: pu.PlotConfig) -> None:
    """Preserve bestfit rendering while composing independent preparation steps"""
    df_minchi2 = read_table(args.minimum_chi_squared)
    binned_groups = LongformatDataFrame.read_from(args.binned_lightcurves).parse(ColumnNames.NU)
    continuous_groups = LongformatDataFrame.read_from(args.candidate_lightcurves).parse(ColumnNames.NU)
    observations = LongformatLightcurve.from_csv(args.obslc)
    parameters = ModelParameters.from_df(read_table(args.model_parameter_table))
    flux = (FluxName.FNU_THERMAL if parameters.switches[SwitchKeys.THERMAL_ONLY]
            else FluxName.FNU_TOTAL)
    fig, axes = plotconf.layout.create()
    try:
        if axes.shape[1] != len(df_minchi2):
            raise ValueError("Bestfit column count must match the number of selected results")
        for col in range(axes.shape[1]):
            result = clc.ChiSquaredResult.from_series(df_minchi2.iloc[col])
            index = integer_identifier(result.model_index)
            nu = QuantityData(result.nu, df_minchi2.attrs[MetadataNames.NU_UNIT])
            observed = observations.select_frequency(nu, unit=observations.metadata.nu_unit)
            df = select_model_df(continuous_groups[result.nu], index, result.nu)
            continuous = build_continuous_lightcurve(df, index, nu, flux)
            df = select_model_df(binned_groups[result.nu], index, result.nu)
            binned = clc.BinnedLightcurveWithBackground.from_df(df, observed.fnu_background.index(0))
            metadata = PlotMetadata.from_model(parameters.select_models([index]), result.chi_squared)
            for row, ax in enumerate(axes[:, col]):
                config = plotconf.layout.axes.for_subplot(row, col)
                plotter = pu.Plotter(ax)
                plot_observed(plotter, observed, col, x_unit=config.x_unit, y_unit=config.y_unit)
                plot_model_curve(plotter, continuous, observed, style_continuous.resolve(col),
                                 x_unit=config.x_unit, y_unit=config.y_unit)
                plot_binned_curve(plotter, binned, observed, style_binned.resolve(col),
                                  x_unit=config.x_unit, y_unit=config.y_unit)
                add_model_metadata(plotter, metadata, plotconf, "")
        sources = {args.obslc, args.plotconfig, args.minimum_chi_squared,
                   args.model_parameter_table, args.candidate_lightcurves, args.binned_lightcurves}
        if args.input is not None:
            sources.add(args.input)
        save_figure(fig, args.output, sources)
    finally:
        plt.close(fig)


SourceKey = Path
ContinuousCurves = dict[CurveKey, dict[FluxName, clc.ContinuousLightcurve]]
BinnedCurves = dict[CurveKey, clc.BinnedLightcurveWithBackground]
ModelMetadata = dict[CurveKey, PlotMetadata]
RunFrequencies = dict[int, list[float]]
LoadedSweep = tuple[list[SweepRun], ContinuousCurves, BinnedCurves, ModelMetadata, RunFrequencies]


def frequency_in_ghz(frequency: QuantityData) -> float:
    """Normalize an explicit positive frequency without assuming its unit"""
    value = real_scalar(frequency.value)
    factor = real_scalar(u.Unit(frequency.unit).to(u.GHz))
    nu = real_scalar(value * factor)
    if nu <= 0:
        raise ValueError("Panel frequency must be positive")
    return nu


def read_sweep_observations(path: Path) -> dict[float, ObservedLightcurve]:
    """Validate observation tables and normalize their frequency identities once"""
    read_table(path)
    lightcurves = LongformatLightcurve.from_csv(path)
    factor = real_scalar(u.Unit(lightcurves.metadata.nu_unit).to(u.GHz))
    observed: dict[float, ObservedLightcurve] = {}
    for raw_nu, observation in lightcurves.group_by_frequency().items():
        nu = real_scalar(real_scalar(raw_nu) * factor)
        if nu <= 0 or any(math.isclose(nu, old, rel_tol=RTOL, abs_tol=0) for old in observed):
            raise ValueError("Nonpositive or ambiguous observation frequencies")
        validate_observed(observation)
        observed[nu] = observation
    if not observed:
        raise ValueError("No observation frequencies are available")
    return observed


def canonical_sweep_sources(
    panels: tuple[SweepPanelSpec, ...],
) -> dict[SourceKey, SweepSource]:
    """Resolve equivalent manifest spellings after rejecting conflicting sources"""
    validate_sweep_panels(panels)
    sources: dict[SourceKey, SweepSource] = {}
    for panel in panels:
        source = panel.source
        key = source.sweep_index.resolve()
        if key not in sources:
            sources[key] = replace(
                source, sweep_index=key,
                minimum_chi_squared_path=normalized_result_path(source.minimum_chi_squared_path),
                model_parameter_table_path=normalized_result_path(source.model_parameter_table_path),
                candidate_lightcurves_path=normalized_result_path(source.candidate_lightcurves_path),
                binned_lightcurves_path=(None if source.binned_lightcurves_path is None
                                        else normalized_result_path(source.binned_lightcurves_path)),
            )
    return sources


def requested_source_frequencies(
    panels: tuple[SweepPanelSpec, ...],
    observed: dict[float, ObservedLightcurve],
) -> list[float]:
    """Deduplicate panel requests using the matched observation frequency keys"""
    return sorted({match_frequency(list(observed), frequency_in_ghz(panel.frequency))
                   for panel in panels})


def validate_axis_sharing(
    panels: tuple[SweepPanelSpec, ...], subplots: pu.SubplotConfig,
) -> None:
    """Reject shared axes whose display units, scales, limits or ticks differ"""
    for direction, sharing in (("x", subplots.sharex), ("y", subplots.sharey)):
        if sharing in (False, "none"):
            continue
        groups: dict[int, pu.AxisConfig] = {}
        for panel in panels:
            group = panel.row if sharing == "row" else panel.col if sharing == "col" else 0
            axis = pu.AxesConfig._single_axis(panel.axes.x if direction == "x" else panel.axes.y)
            previous = groups.get(group)
            if previous is not None and (
                u.Unit(previous.label.unit) != u.Unit(axis.label.unit)
                or previous.scale != axis.scale or previous.limits != axis.limits
                or previous.major_ticks != axis.major_ticks or previous.minor_ticks != axis.minor_ticks
            ):
                raise ValueError(f"Incompatible shared {direction} axes at panel {(panel.row, panel.col)}")
            groups[group] = axis


def load_sweep_source(
    source: SweepSource,
    *,
    observed: dict[float, ObservedLightcurve],
    components: tuple[FluxName, ...],
    show_binned: bool,
    requested_frequencies: list[float] | None,
    source_files: set[Path],
) -> LoadedSweep:
    """Validate one source and construct reusable selected curves without Axes"""
    validate_sweep_source(source)
    supported = (FluxName.FNU_TOTAL, FluxName.FNU_THERMAL, FluxName.FNU_POWERLAW)
    if not components or len(set(components)) != len(components) or any(c not in supported for c in components):
        raise ValueError("Select distinct supported continuous flux components")
    parameter = source.parameter
    binned_path = source.binned_lightcurves_path
    if show_binned and binned_path is None:
        raise ValueError("Selected binned curves require a source binned path")
    runs = read_sweep_runs(source.sweep_index, parameter=parameter)
    source_files.add(source.sweep_index)
    curves: dict[CurveKey, dict[FluxName, clc.ContinuousLightcurve]] = {}
    binned_curves: dict[CurveKey, clc.BinnedLightcurveWithBackground] = {}
    metadata: dict[CurveKey, PlotMetadata] = {}
    frequencies: dict[int, list[float]] = {}
    for run in runs:
        source_files.add(run.input_path)
        tables: dict[str, pd.DataFrame] = {}
        paths: dict[str, Path] = {}
        for name, relative in (("minimum", source.minimum_chi_squared_path),
                               ("parameters", source.model_parameter_table_path),
                               ("continuous", source.candidate_lightcurves_path)):
            path = run_result_path(run, relative)
            paths[name] = path
            source_files.add(path)
            tables[name] = read_table(path)
            validate_provenance(tables[name], run, path)
        if show_binned:
            if binned_path is None:
                raise ValueError("Binned result path is required")
            path = run_result_path(run, binned_path)
            source_files.add(path)
            tables["binned"] = read_table(path)
            validate_provenance(tables["binned"], run, path)
        validate_powerlaw_metadata(tables["parameters"], paths["parameters"])
        parameters = ModelParameters.from_df(tables["parameters"])
        if parameters.switches[SwitchKeys.THERMAL_ONLY]:
            raise ValueError(f"Thermal-only result parameters: {run.run_name}")
        # Both attrs are required even when the curve tables omit power-law parameters
        if not math.isclose(model_sweep_value(parameters, parameter), run.sweep_value,
                            rel_tol=RTOL, abs_tol=0):
            raise ValueError(
                f"Parameter-table {parameter} mismatch: {run.run_name}: {paths['parameters']}"
            )
        config_input = build_input.load_yaml(run.input_path)
        expected_distance = real_scalar(build_input.get_nested_value(config_input, "model_parameters.distance.value"))
        distance_unit: object = build_input.get_nested_value(config_input, "model_parameters.distance.unit")
        if not isinstance(distance_unit, str):
            raise ValueError(f"Invalid input distance unit: {run.input_path}")
        if not math.isclose(real_scalar(parameters.distance.value_in(distance_unit)), expected_distance,
                            rel_tol=RTOL, abs_tol=0):
            raise ValueError(f"Parameter-table distance mismatch: {run.run_name}")
        for column in (ColumnNames.PHI_TH, ColumnNames.THETA0):
            if np.any(numeric_values(tables["parameters"], column) <= 0):
                raise ValueError(f"Nonpositive model parameter {column}: {run.run_name}")
        minima = frequency_groups(tables["minimum"])
        continuous_groups = frequency_groups(tables["continuous"])
        binned_groups = frequency_groups(tables["binned"]) if show_binned else {}
        for group in continuous_groups.values():
            validate_samples(group)
            for component in components:
                numeric_values(group, component)
        for group in binned_groups.values():
            validate_samples(group, binned=True)
        frequencies[run.run_id] = sorted(minima)
        if len(select_frequencies(sorted(minima), sorted(minima))) != len(minima):
            raise ValueError(f"Ambiguous bestfit frequencies: {run.run_name}")
        for nu, df_min in minima.items():
            try:
                if len(df_min) != 1:
                    raise ValueError("Expected exactly one bestfit result per frequency")
                index = integer_identifier(require_series(df_min, ColumnNames.MODEL_INDEX).iloc[0])
                result = clc.ChiSquaredResult.from_series(df_min.iloc[0])
                chi2 = real_scalar(result.chi_squared)
                if chi2 < 0:
                    raise ValueError("Negative chi-squared")
                obs_nu = match_frequency(list(observed), nu)
                obs = observed[obs_nu]
                nu_quantity = QuantityData(nu, "GHz")
                source_nu = match_frequency(list(continuous_groups), nu)
                df_continuous = select_model_df(continuous_groups[source_nu], index, source_nu)
                selected = requested_frequencies is None or any(
                    math.isclose(obs_nu, nu_requested, rel_tol=RTOL, abs_tol=0)
                    for nu_requested in requested_frequencies
                )
                if not selected:
                    if show_binned:
                        bin_nu = match_frequency(list(binned_groups), nu)
                        select_model_df(binned_groups[bin_nu], index, bin_nu)
                    continue
                key = (run.run_id, obs_nu, index)
                curves[key] = {component: build_continuous_lightcurve(
                    df_continuous, index, nu_quantity, component,
                ) for component in components}
                metadata[key] = PlotMetadata.from_model(parameters.select_models([index]), chi2)
                if show_binned:
                    bin_nu = match_frequency(list(binned_groups), nu)
                    df_bin = select_model_df(binned_groups[bin_nu], index, bin_nu)
                    binned = clc.BinnedLightcurveWithBackground.from_df(df_bin, obs.fnu_background.index(0))
                    values = np.asarray(binned.fnu_with_bg.to_value_in("mJy"), dtype=float)
                    if not np.all(np.isfinite(values)) or values.size != len(df_bin):
                        raise ValueError("Invalid binned flux values")
                    binned_curves[key] = binned
            except (KeyError, ValueError, TypeError) as error:
                raise ValueError(f"{run.run_name}, nu={nu:g} GHz: {error}") from error
    reference = frequencies[runs[0].run_id]
    if requested_frequencies is None:
        for run in runs:
            available = frequencies[run.run_id]
            if len(available) != len(reference) or select_frequencies(available, reference) != available:
                raise ValueError(f"Bestfit frequency sets differ: {run.run_name}")
        if len(observed) != len(reference):
            raise ValueError("Observation and bestfit frequency sets differ; use --frequencies explicitly")
    selected = select_frequencies(reference, requested_frequencies)
    for run in runs:
        select_frequencies(frequencies[run.run_id], selected)
    return runs, curves, binned_curves, metadata, frequencies


def draw_sweep_panel(
    ax: Axes,
    panel: SweepPanelSpec,
    observed: ObservedLightcurve,
    *,
    nu_ghz: float,
    runs: list[SweepRun],
    curves: ContinuousCurves,
    binned: BinnedCurves,
    metadata: ModelMetadata,
    colors: dict[int, str],
    metadata_config: pu.MetadataGroupConfig,
) -> None:
    """Draw one source's already prepared curves into an existing Axes"""
    axis = panel.axes
    plotter = pu.Plotter(ax)
    check_axes(axis, observed.t.to_value_in(axis.x_unit), observed.fnu_total.to_value_in(axis.y_unit))
    check_axes(axis, observed.t.to_value_in(axis.x_unit), observed.fnu_background.to_value_in(axis.y_unit))
    plot_observed(plotter, observed, panel.col, x_unit=axis.x_unit, y_unit=axis.y_unit, sweep=True)
    keys = [key for key in curves if key[1] == nu_ghz]
    if len(keys) != len(runs) or {key[0] for key in keys} != {run.run_id for run in runs}:
        raise ValueError(f"Incomplete prepared curves at panel {(panel.row, panel.col)}")
    title = [pu.QuantityLabel(r"\nu", ".4g", r"\mathrm{GHz}").to_text(nu_ghz)]
    for name in sweep_title_fields(panel.source.parameter):
        if name not in metadata_config.items:
            continue
        quantities = [metadata[key].to_dict()[name] for key in keys]
        quantity = quantities[0]
        value = real_scalar(quantity.value)
        if any(not math.isclose(real_scalar(other.value_in(quantity.unit)), value,
                                rel_tol=RTOL, abs_tol=0) for other in quantities):
            raise ValueError(f"Shared title field varies: {name}, panel {(panel.row, panel.col)}")
        pu.MetadataComponent(quantity, metadata_config.items[name]).add_strings(title)
    plotter.title(strings=title, fontsize=metadata_config.fontsize)
    component_styles = dict(panel.settings.component_styles)
    for key in keys:
        for component in panel.settings.components:
            continuous = curves[key][component]
            style = replace(component_styles[component], color=colors[key[0]])
            check_axes(axis, continuous.t.to_value_in(axis.x_unit),
                       continuous.fnu_with_bg(observed.fnu_background.index(0)).to_value_in(axis.y_unit))
            plot_model_curve(plotter, continuous, observed, style,
                             x_unit=axis.x_unit, y_unit=axis.y_unit)
        if panel.settings.show_binned:
            curve = binned[key]
            check_axes(axis, curve.bin_edges.to_value_in(axis.x_unit),
                       curve.fnu_with_bg.to_value_in(axis.y_unit))
            plot_binned_curve(plotter, curve, observed,
                              replace(style_binned.resolve(0), color=colors[key[0]]),
                              x_unit=axis.x_unit, y_unit=axis.y_unit)
    label = SWEEP_LABELS[panel.source.parameter]
    handles = [Line2D([], [], color=colors[run.run_id], label=label.to_text(run.sweep_value))
               for run in sorted(runs, key=lambda run: run.sweep_value)]
    ax.legend(handles=handles, fontsize=18)
    print(f"Panel {(panel.row, panel.col)}, {panel.source.sweep_index}, {nu_ghz:g} GHz: "
          f"{len(keys)} model curves; {panel.source.parameter}={[run.sweep_value for run in runs]}",
          flush=True)


def render_prepared_sweep(
    panels: tuple[SweepPanelSpec, ...],
    *,
    observed: dict[float, ObservedLightcurve],
    runs_by_source: dict[SourceKey, list[SweepRun]],
    curves_by_source: dict[SourceKey, ContinuousCurves],
    binned_by_source: dict[SourceKey, BinnedCurves],
    metadata_by_source: dict[SourceKey, ModelMetadata],
    plotconf: pu.PlotConfig,
    output: Path,
    source_files: set[Path],
) -> None:
    """Compose one Figure from source-separated prepared data and publish once"""
    nrows, ncols = validate_sweep_panels(panels)
    subplot = replace(plotconf.layout.subplots, nrows=nrows, ncols=ncols)
    validate_axis_sharing(panels, subplot)
    frequencies = [match_frequency(list(observed), frequency_in_ghz(panel.frequency)) for panel in panels]
    colors: dict[tuple[SourceKey, str], dict[int, str]] = {}
    for panel in panels:
        source_key = panel.source.sweep_index.resolve()
        color_key = (source_key, panel.settings.cmap)
        if color_key not in colors:
            colors[color_key] = color_mapping(runs_by_source[source_key], panel.settings)
    figure_config = plotconf.layout.figure
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figure_config.figsize, dpi=figure_config.dpi,
        sharex=subplot.sharex, sharey=subplot.sharey, layout=figure_config.layout_engine,
        squeeze=False,
    )
    try:
        for panel, nu in zip(panels, frequencies):
            ax = axes[panel.row, panel.col]
            pu.AxesConfigurator.apply(ax, panel.axes)
            source_key = panel.source.sweep_index.resolve()
            draw_sweep_panel(
                ax, panel, observed[nu], nu_ghz=nu,
                runs=runs_by_source[source_key], curves=curves_by_source[source_key],
                binned=binned_by_source[source_key], metadata=metadata_by_source[source_key],
                colors=colors[(source_key, panel.settings.cmap)], metadata_config=plotconf.metadata,
            )
        if subplot.hide_shared_labels:
            for panel in panels:
                ax = axes[panel.row, panel.col]
                if ((subplot.sharex in (True, "all", "col") and panel.row < nrows - 1)
                        or (subplot.sharex == "row" and panel.col > 0)):
                    ax.set_xlabel("")
                if ((subplot.sharey in (True, "all", "row") and panel.col > 0)
                        or (subplot.sharey == "col" and panel.row > 0)):
                    ax.set_ylabel("")
        save_figure(fig, output, source_files)
    finally:
        plt.close(fig)


def draw_sweep(
    panels: tuple[SweepPanelSpec, ...],
    *,
    obslc: Path,
    plotconf: pu.PlotConfig,
    output: Path,
    plotconfig_path: Path | None = None,
) -> None:
    """Prepare independent sources and render explicitly placed Python panels"""
    sources = canonical_sweep_sources(panels)
    validate_axis_sharing(panels, plotconf.layout.subplots)
    observed = read_sweep_observations(obslc)
    source_files = {obslc}
    if plotconfig_path is not None:
        source_files.add(plotconfig_path)
    runs_by_source: dict[SourceKey, list[SweepRun]] = {}
    curves_by_source: dict[SourceKey, ContinuousCurves] = {}
    binned_by_source: dict[SourceKey, BinnedCurves] = {}
    metadata_by_source: dict[SourceKey, ModelMetadata] = {}
    frequencies_by_source: dict[SourceKey, RunFrequencies] = {}
    for key, source in sources.items():
        consumers = tuple(panel for panel in panels if panel.source.sweep_index.resolve() == key)
        components = tuple(dict.fromkeys(component for panel in consumers
                                         for component in panel.settings.components))
        wanted = requested_source_frequencies(consumers, observed)
        try:
            (runs_by_source[key], curves_by_source[key], binned_by_source[key],
             metadata_by_source[key], frequencies_by_source[key]) = load_sweep_source(
                source, observed=observed, components=components,
                show_binned=any(panel.settings.show_binned for panel in consumers),
                requested_frequencies=wanted, source_files=source_files,
            )
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise ValueError(f"Source {key}, parameter={source.parameter}: {error}") from error
    render_prepared_sweep(
        panels, observed=observed, runs_by_source=runs_by_source,
        curves_by_source=curves_by_source, binned_by_source=binned_by_source,
        metadata_by_source=metadata_by_source, plotconf=plotconf,
        output=output, source_files=source_files,
    )


def draw_legacy_sweep(args: argparse.Namespace, plotconf: pu.PlotConfig) -> None:
    """Adapt existing CLI inputs to the same source-separated panel renderer"""
    source = SweepSource(
        validate_sweep_parameter(args.parameter), args.sweep_index,
        args.minimum_chi_squared_path, args.model_parameter_table_path,
        args.candidate_lightcurves_path, args.binned_lightcurves_path,
    )
    settings = replace(build_sweep_settings(), show_binned=source.binned_lightcurves_path is not None)
    observed = read_sweep_observations(args.obslc)
    source_files = {args.obslc, args.plotconfig}
    runs, curves, binned, metadata, frequencies = load_sweep_source(
        source, observed=observed, components=settings.components,
        show_binned=settings.show_binned, requested_frequencies=args.frequencies,
        source_files=source_files,
    )
    reference = frequencies[runs[0].run_id]
    selected = select_frequencies(reference, args.frequencies)
    axes = plotconf.layout.axes
    if isinstance(axes.x, tuple):
        if len(axes.x) != len(reference):
            raise ValueError("Per-frequency x-axis settings must match the reference frequency set")
        axes = replace(axes, x=tuple(axes.x[reference.index(match_frequency(reference, nu))]
                                    for nu in selected))
    if isinstance(axes.y, tuple) and len(axes.y) != 1:
        raise ValueError("Sweep layout requires one y-axis configuration row")
    panels = build_sweep_panels(
        rows=((source, settings),), frequencies=tuple(QuantityData(nu, "GHz") for nu in selected),
        axes=axes,
    )
    key = source.sweep_index.resolve()
    render_prepared_sweep(
        panels, observed=observed, runs_by_source={key: runs}, curves_by_source={key: curves},
        binned_by_source={key: binned}, metadata_by_source={key: metadata},
        plotconf=plotconf, output=args.output, source_files=source_files,
    )

@timed
def main() -> None:
    args = parse_args()
    if args.mode == "candidates":
        raise NotImplementedError("candidatesモードの描画処理は未実装です")
    plotconf = pu.PlotConfig.from_yaml(args.plotconfig)
    if args.mode == "sweep":
        draw_legacy_sweep(args, plotconf)
    else:
        draw_bestfit(args, plotconf)
    print(f"Saved lightcurves: {args.output}", flush=True)


if __name__ == "__main__":
    main()
