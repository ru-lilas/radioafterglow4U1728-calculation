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

try:
    from . import build_input
except ImportError:
    import build_input

style_continuous = pu.LineStyleConfig(
    colors=[
        "#FF4B00",
        "#005AFF"
    ],
    linestyle="--",
    linewidth=2.0
)
style_thermal = replace(
    style_continuous,
    linestyle=":",
)
style_binned = replace(
    style_continuous,
    linestyle="-",
    linewidth=2.0
)
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
    parser.add_argument("--parameter", choices=("f_e",))
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
    f_e: float


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


def read_sweep_runs(path: Path) -> list[SweepRun]:
    """Validate manifest identity and full input family before loading results"""
    expected = build_input.SWEEP_SPECS["f_e"].manifest_fields
    runs: list[SweepRun] = []
    families: list[object] = []
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != expected:
            raise ValueError(f"Expected f_e manifest columns {expected}: {path}")
        for row in reader:
            if None in row or any(value is None or value == "" for value in row.values()):
                raise ValueError(f"Malformed manifest record: {row}")
            # CSV identity fields are textual by definition, unlike numeric result columns
            run_id = integer_identifier(int(row["run_id"]), minimum=1)
            f_e = build_input.validate_sweep_values([float(row["f_e"])], parameter="f_e")[0]
            run = SweepRun(run_id, row["run_name"], Path(row["input_path"]), row["config_hash"], f_e)
            if run.run_name != f"run_{run_id:06d}" or run.input_path.parent.name != run.run_name:
                raise ValueError(f"Conflicting RUN identity: {row}")
            if row["status"] not in ("created", "reused"):
                raise ValueError(f"Invalid status for {run.run_name}")
            if any(old.run_id == run_id or old.input_path.resolve() == run.input_path.resolve()
                   or math.isclose(old.f_e, f_e, rel_tol=RTOL, abs_tol=0) for old in runs):
                raise ValueError(f"Duplicate or ambiguous sweep assignment: {row}")
            config = build_input.load_yaml(run.input_path)
            build_input.validate_f_e_input(config)
            actual = real_scalar(build_input.get_nested_value(config, build_input.SWEEP_SPECS["f_e"].yaml_path))
            if not math.isclose(actual, f_e, rel_tol=RTOL, abs_tol=0):
                raise ValueError(f"Manifest f_e disagrees with {run.input_path}")
            if build_input.calculate_config_hash(config) != run.config_hash:
                raise ValueError(f"Manifest hash disagrees with {run.input_path}")
            family = build_input.sweep_family(config, parameter="f_e")
            if families and family != families[0]:
                raise ValueError(f"Fixed input settings differ: {run.input_path}")
            families.append(family)
            runs.append(run)
    if not runs:
        raise ValueError(f"Empty sweep manifest: {path}")
    return sorted(runs, key=lambda run: run.f_e)


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
                key = f"model_parameters.shock.powerlaw.{name}"
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
    """Assign evenly spaced colormap colors in ascending f_e order"""
    ordered = sorted(runs, key=lambda run: run.f_e)
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


def draw_sweep(args: argparse.Namespace, plotconf: pu.PlotConfig) -> None:
    """Select each RUN's bestfit independently and compose a shared-color figure"""
    settings = build_sweep_settings()
    if (not settings.components or len(set(settings.components)) != len(settings.components)
            or any(name not in (FluxName.FNU_TOTAL, FluxName.FNU_THERMAL, FluxName.FNU_POWERLAW)
                   for name in settings.components)):
        raise ValueError("Select distinct supported continuous flux components")
    component_styles = dict(settings.component_styles)
    if len(component_styles) != len(settings.component_styles) or any(
        component not in component_styles for component in settings.components
    ):
        raise ValueError("Configure one style per selected flux component")
    binned_path: Path | None = args.binned_lightcurves_path
    if settings.show_binned and binned_path is None:
        raise ValueError("Selected binned curves require --binned-lightcurves-path")
    if not settings.show_binned and binned_path is not None:
        raise ValueError("Enable show_binned in build_sweep_settings() to use binned results")
    runs = read_sweep_runs(args.sweep_index)
    colors = color_mapping(runs, settings)
    sources: set[Path] = {args.sweep_index, args.obslc, args.plotconfig}
    # Check original observation headers, then reuse the existing observation abstraction
    read_table(args.obslc)
    lcs_obs = LongformatLightcurve.from_csv(args.obslc)
    obs_factor = real_scalar(u.Unit(lcs_obs.metadata.nu_unit).to(u.GHz))
    observed: dict[float, ObservedLightcurve] = {}
    for raw_nu, obs in lcs_obs.group_by_frequency().items():
        nu = real_scalar(raw_nu) * obs_factor
        if nu <= 0 or any(math.isclose(nu, value, rel_tol=RTOL, abs_tol=0) for value in observed):
            raise ValueError("Nonpositive or ambiguous observation frequencies")
        validate_observed(obs)
        observed[nu] = obs
    curves: dict[CurveKey, dict[FluxName, clc.ContinuousLightcurve]] = {}
    binned_curves: dict[CurveKey, clc.BinnedLightcurveWithBackground] = {}
    metadata: dict[CurveKey, PlotMetadata] = {}
    frequencies: dict[int, list[float]] = {}
    for run in runs:
        sources.add(run.input_path)
        tables: dict[str, pd.DataFrame] = {}
        paths: dict[str, Path] = {}
        for name, relative in (("minimum", args.minimum_chi_squared_path),
                               ("parameters", args.model_parameter_table_path),
                               ("continuous", args.candidate_lightcurves_path)):
            path = run_result_path(run, relative)
            paths[name] = path
            sources.add(path)
            tables[name] = read_table(path)
            validate_provenance(tables[name], run, path)
        if settings.show_binned:
            if binned_path is None:
                raise ValueError("Binned result path is required")
            path = run_result_path(run, binned_path)
            sources.add(path)
            tables["binned"] = read_table(path)
            validate_provenance(tables["binned"], run, path)
        parameters = ModelParameters.from_df(tables["parameters"])
        if parameters.switches[SwitchKeys.THERMAL_ONLY]:
            raise ValueError(f"Thermal-only result parameters: {run.run_name}")
        # The parameter-table schema must carry f_e even if curve files omit it
        if not math.isclose(real_scalar(parameters.powerlaw.f_e), run.f_e, rel_tol=RTOL, abs_tol=0):
            raise ValueError(f"Parameter-table f_e mismatch: {run.run_name}")
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
        binned_groups = frequency_groups(tables["binned"]) if settings.show_binned else {}
        for group in continuous_groups.values():
            validate_samples(group)
            for component in settings.components:
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
                key = (run.run_id, obs_nu, index)
                curves[key] = {component: build_continuous_lightcurve(
                    df_continuous, index, nu_quantity, component,
                ) for component in settings.components}
                metadata[key] = PlotMetadata.from_model(parameters.select_models([index]), chi2)
                if settings.show_binned:
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
    if args.frequencies is None:
        for run in runs:
            available = frequencies[run.run_id]
            if len(available) != len(reference) or select_frequencies(available, reference) != available:
                raise ValueError(f"Bestfit frequency sets differ: {run.run_name}")
        if len(observed) != len(reference):
            raise ValueError("Observation and bestfit frequency sets differ; use --frequencies explicitly")
    selected = select_frequencies(reference, args.frequencies)
    for run in runs:
        select_frequencies(frequencies[run.run_id], selected)
    selected_obs = [match_frequency(list(observed), nu) for nu in selected]
    layout = plotconf.layout
    if isinstance(layout.axes.x, tuple):
        if len(layout.axes.x) != len(reference):
            raise ValueError("Per-frequency x-axis configuration must match the reference frequency set")
        positions = [reference.index(match_frequency(reference, nu)) for nu in selected]
        layout = replace(layout, axes=replace(layout.axes, x=tuple(layout.axes.x[i] for i in positions)))
    if isinstance(layout.axes.y, tuple) and len(layout.axes.y) != 1:
        raise ValueError("Sweep layout requires one y-axis configuration row")
    layout = replace(layout, subplots=replace(layout.subplots, nrows=1, ncols=len(selected)))
    fig, axes = layout.create()
    try:
        for col, obs_nu in enumerate(selected_obs):
            ax = axes[0, col]
            axis = layout.axes.for_subplot(0, col)
            obs = observed[obs_nu]
            plotter = pu.Plotter(ax)
            check_axes(axis, obs.t.to_value_in(axis.x_unit), obs.fnu_total.to_value_in(axis.y_unit))
            check_axes(axis, obs.t.to_value_in(axis.x_unit), obs.fnu_background.to_value_in(axis.y_unit))
            plot_observed(plotter, obs, col, x_unit=axis.x_unit, y_unit=axis.y_unit, sweep=True)
            keys = [key for key in curves if key[1] == obs_nu]
            title = [pu.QuantityLabel(r"\nu", ".4g", r"\mathrm{GHz}").to_text(obs_nu)]
            for name in ("eps_b", "eps_th", "mu_e_mu", "distance", "p_index"):
                if name not in plotconf.metadata.items:
                    continue
                quantities = [metadata[key].to_dict()[name] for key in keys]
                quantity = quantities[0]
                value = real_scalar(quantity.value)
                if any(not math.isclose(real_scalar(other.value_in(quantity.unit)), value,
                                        rel_tol=RTOL, abs_tol=0) for other in quantities):
                    raise ValueError(f"Shared title field varies: {name}")
                pu.MetadataComponent(quantity, plotconf.metadata.items[name]).add_strings(title)
            plotter.title(strings=title, fontsize=plotconf.metadata.fontsize)
            for key in keys:
                for component, continuous in curves[key].items():
                    style = replace(component_styles[component], color=colors[key[0]])
                    check_axes(axis, continuous.t.to_value_in(axis.x_unit),
                               continuous.fnu_with_bg(obs.fnu_background.index(0)).to_value_in(axis.y_unit))
                    plot_model_curve(plotter, continuous, obs, style,
                                     x_unit=axis.x_unit, y_unit=axis.y_unit)
                if settings.show_binned:
                    binned = binned_curves[key]
                    check_axes(axis, binned.bin_edges.to_value_in(axis.x_unit),
                               binned.fnu_with_bg.to_value_in(axis.y_unit))
                    plot_binned_curve(plotter, binned, obs, replace(style_binned.resolve(0), color=colors[key[0]]),
                                      x_unit=axis.x_unit, y_unit=axis.y_unit)
            print(f"{obs_nu:g} GHz: {len(keys)} model curves; f_e={[run.f_e for run in runs]}", flush=True)
            handles = [
                Line2D(
                    [], [],
                    color=colors[run.run_id],
                    label=pu.QuantityLabel(
                        r"f_{\mathrm{e}}", ".1e", "",
                    ).to_text(run.f_e),
                )
                for run in sorted(runs, key=lambda run: run.f_e)
            ]
            axes.flat[col].legend(handles=handles,fontsize=18)

            save_figure(fig, args.output, sources)
        save_figure(fig, args.output, sources)
    finally:
        plt.close(fig)


@timed
def main() -> None:
    args = parse_args()
    if args.mode == "candidates":
        raise NotImplementedError("candidatesモードの描画処理は未実装です")
    plotconf = pu.PlotConfig.from_yaml(args.plotconfig)
    if args.mode == "sweep":
        draw_sweep(args, plotconf)
    else:
        draw_bestfit(args, plotconf)
    print(f"Saved lightcurves: {args.output}", flush=True)


if __name__ == "__main__":
    main()
