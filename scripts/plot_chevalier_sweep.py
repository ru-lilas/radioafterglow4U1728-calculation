"""Compare manifest-selected RUNs as columns and Chevalier plot quantities as rows"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure

from module import plot_utils as pu
from module.chevalier_plotting import (
    ChevalierInputPaths, ChevalierPlotData, ColormapRowConfig, ContourRowConfig,
    QUANTITY_RESOLVERS, RowConfig, colormap_row, colormap_values, contour_row, draw_colormap_panel,
    draw_contour_panel, plane_signature, shared_plane_axes, visible_values,
)

if __package__:
    from .build_input import SWEEP_SPECS, calculate_config_hash, get_nested_value, load_yaml, sweep_family
    from .combine_energetics import RunEntry, read_manifest, real_scalar
else:
    from build_input import SWEEP_SPECS, calculate_config_hash, get_nested_value, load_yaml, sweep_family
    from combine_energetics import RunEntry, read_manifest, real_scalar


@dataclass(frozen=True, slots=True)
class SweepSettings:
    rows: tuple[RowConfig, ...]
    base_config: pu.PlotConfig
    columns_per_page: int = 3
    cell_size: tuple[float, float] = (4.5, 3.7)
    dpi: int = 120
    title_fontsize: float = 12


@dataclass(frozen=True, slots=True)
class SelectedRun:
    entry: RunEntry
    paths: ChevalierInputPaths
    fingerprint: str
    eps_b: float
    eps_th: float

    def title(self) -> str:
        """Separate RUN and sweep parameters using comma and one ASCII space"""
        return ", ".join((self.entry.run_name, f"{self.entry.parameter}={self.entry.value:g}"))


@dataclass(frozen=True, slots=True)
class SweepPlan:
    runs: tuple[SelectedRun, ...]
    settings: SweepSettings
    protected: tuple[Path, ...]


def build_plot_config(contours: pu.PlotConfig, maps: pu.PlotConfig) -> SweepSettings:
    """Edit row instances, their order, pagination and cell sizes here"""
    if plane_signature(shared_plane_axes(contours)) != plane_signature(shared_plane_axes(maps)):
        raise ValueError("Contour and colormap configurations must use the same Chevalier plane")
    contour = contour_row(contours)
    wind = replace(colormap_row(maps, "a_wind"), label="A_wind", axes=contour.axes)
    beta = replace(colormap_row(maps, "beta_sh"), label="beta_sh", axes=contour.axes)
    ratio = ColormapRowConfig(
        quantity="xm_peak_cutoff_ratio", label="log10(Peak / cutoff)", axes=contour.axes,
        transform="log10",
        style=replace(
            wind.style,
            cmap="coolwarm",
            norm=pu.NormConfig(
                "centered",
                vcenter=0.0
            ),
            colorbar=replace(wind.style.colorbar,
                label=pu.LabelConfigure(r"$\log_{10}(x_{M,\mathrm{peak}}/x_{M,\mathrm{cutoff}})$", "", 18))),
    )
    # Add, remove or reorder complete row instances here
    rows = (contour, wind, beta, ratio)
    return SweepSettings(rows=rows, base_config=contours)


def validate_settings(settings: SweepSettings) -> None:
    """Validate all selected rows before opening any RUN data"""
    if not settings.rows:
        raise ValueError("Configure at least one sweep row")
    if isinstance(settings.columns_per_page, bool) or not isinstance(settings.columns_per_page, int) or settings.columns_per_page < 1:
        raise ValueError("columns_per_page must be a positive integer")
    for size in settings.cell_size:
        if real_scalar(size) <= 0:
            raise ValueError("Cell dimensions must be positive")
    if isinstance(settings.dpi, bool) or not isinstance(settings.dpi, int) or settings.dpi < 1:
        raise ValueError("dpi must be a positive integer")
    if real_scalar(settings.title_fontsize) <= 0:
        raise ValueError("Title fontsize must be positive")
    signature = plane_signature(shared_plane_axes(settings.base_config))
    for row in settings.rows:
        if not isinstance(row.label, str) or not row.label.strip():
            raise ValueError("Every row requires a nonempty label")
        if plane_signature(row.axes) != signature:
            raise ValueError("Every sweep row must use the same coordinate system")
        if isinstance(row, ColormapRowConfig):
            quantities = (row.quantity,) + tuple(component.quantity for component in row.contours)
            if any(component.style.levels.values is None for component in row.contours):
                raise ValueError("Colormap overlay contours require explicit shared levels")
            center = 0.0 if row.transform == "log10" else 1.0
            if row.quantity == "xm_peak_cutoff_ratio" and row.style.norm.type == "centered" and row.style.norm.vcenter != center:
                raise ValueError(f"The ratio colormap requires vcenter={center:g} for centered normalization")
            if row.style.norm.type not in ("linear", "log", "centered"):
                raise ValueError(f"Unsupported normalization for {row.label}")
        elif isinstance(row, ContourRowConfig):
            if not row.components:
                raise ValueError("Contour rows require at least one contour component")
            quantities = tuple(component.quantity for component in row.components)
        else:
            raise TypeError("Unsupported sweep row configuration")
        for name in quantities:
            if name not in QUANTITY_RESOLVERS:
                raise ValueError(f"Undefined quantity {name}; provide its scientific data contract and resolver")


def within_run(entry: RunEntry, relative: Path) -> Path:
    """Resolve an explicit relative data path strictly inside its RUN"""
    root = entry.input_path.parent.resolve()
    path = root / relative
    if relative.is_absolute() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Data path must remain within {entry.run_name}: {relative}")
    return path


def select_runs(entries: list[RunEntry], *, peak_path: Path, model_path: Path,
                obslc: Path, bestfit_path: Path | None, skip_missing: bool) -> tuple[SelectedRun, ...]:
    """Verify one sweep family and select complete manifest-listed RUNs only"""
    if not entries:
        raise ValueError("The sweep manifest is empty")
    if not obslc.is_file():
        raise FileNotFoundError(f"Shared observation file is unavailable: {obslc}")
    parameter = entries[0].parameter
    if any(entry.parameter != parameter for entry in entries):
        raise ValueError("Mixed sweep parameters in manifest entries")
    family: object = None
    selected: list[SelectedRun] = []
    for entry in entries:
        try:
            config = load_yaml(entry.input_path)
            current = sweep_family(config, parameter=parameter)
            if family is None:
                family = current
            elif current != family:
                raise ValueError("Input belongs to a different sweep family")
            fingerprint = calculate_config_hash(config)
            if entry.config_hash is not None and entry.config_hash != fingerprint:
                raise ValueError("Manifest hash disagrees with input")
            eps_b = real_scalar(get_nested_value(config, SWEEP_SPECS["eps_b"].yaml_path))
            eps_th = real_scalar(get_nested_value(config, SWEEP_SPECS["eps_th"].yaml_path))
            if not 0 < eps_b <= 1 or not 0 < eps_th <= 1:
                raise ValueError("Input energy fractions must be in (0, 1]")
            value = eps_b if parameter == "eps_b" else eps_th
            if not math.isclose(value, entry.value, rel_tol=1e-10, abs_tol=0):
                raise ValueError("Manifest sweep value disagrees with input")
            paths = ChevalierInputPaths(
                within_run(entry, peak_path), within_run(entry, model_path), obslc,
                within_run(entry, bestfit_path) if bestfit_path is not None else None,
            )
            missing = [path for path in paths.sources() if not path.exists()]
            if missing:
                if skip_missing:
                    print(f"WARNING: {entry.run_name}: entire RUN skipped; missing {missing}", flush=True)
                    continue
                raise FileNotFoundError(f"Missing RUN CSVs: {missing}")
            if any(not path.is_file() for path in paths.sources()):
                raise ValueError("Each CSV source must be a regular file")
            selected.append(SelectedRun(entry, paths, fingerprint, eps_b, eps_th))
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{entry.run_name}: {error}") from error
    if not selected:
        raise ValueError("No complete RUN remains")
    print(f"Included RUNs: {len(selected)}; skipped RUNs: {len(entries)-len(selected)}", flush=True)
    return tuple(selected)


def load_run(run: SelectedRun, settings: SweepSettings, *, report_missing_hashes: bool) -> ChevalierPlotData:
    """Load one RUN and verify model parameters and any available result hashes"""
    try:
        if calculate_config_hash(load_yaml(run.entry.input_path)) != run.fingerprint:
            raise ValueError("Input changed during sweep plotting")
        data = ChevalierPlotData.from_paths(run.paths, settings.base_config)
        for actual, expected, name in ((data.model_parameters.eps_b, run.eps_b, "eps_b"),
                                       (data.model_parameters.eps_th, run.eps_th, "eps_th")):
            if not math.isclose(real_scalar(actual), expected, rel_tol=1e-10, abs_tol=0):
                raise ValueError(f"Model-table {name} disagrees with input")
        missing: list[str] = []
        for source in data.provenance:
            if not source.hashes:
                missing.append(str(source.path))
            for fingerprint in source.hashes:
                if not isinstance(fingerprint, str) or fingerprint != run.fingerprint:
                    raise ValueError(f"Result hash disagrees with input: {source.path}")
        if report_missing_hashes and missing:
            print(f"WARNING: {run.entry.run_name}: result provenance incomplete; stale results cannot be excluded: {missing}", flush=True)
        return data
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{run.entry.run_name}: {error}") from error


def extend_range(current: tuple[float, float] | None, values: NDArray[np.float64]) -> tuple[float, float] | None:
    """Reduce one visible array without retaining a RUN's grid"""
    if values.size == 0:
        return current
    lower, upper = real_scalar(values.min()), real_scalar(values.max())
    return (lower, upper) if current is None else (min(current[0], lower), max(current[1], upper))


def resolve_rows(runs: tuple[SelectedRun, ...], settings: SweepSettings) -> SweepSettings:
    """Resolve global color bounds and contour levels with one RUN in memory"""
    validate_settings(settings)
    keys: list[tuple[int, int]] = []
    for index, row in enumerate(settings.rows):
        keys.extend((index, item) for item in range(len(row.components) if isinstance(row, ContourRowConfig) else 1))
    ranges: dict[tuple[int, int], tuple[float, float] | None] = dict.fromkeys(keys)
    for run in runs:
        data = load_run(run, settings, report_missing_hashes=True)
        for index, row in enumerate(settings.rows):
            components = row.components if isinstance(row, ContourRowConfig) else ()
            items = ((component.quantity, component.style.unit, component.style.levels.scale)
                     for component in components)
            if isinstance(row, ColormapRowConfig):
                samples = ((row.quantity, row.style.colorbar.label.unit, row.style.norm.type),)
            else:
                samples = tuple(items)
            for item, (quantity, unit, normalization) in enumerate(samples):
                values = (colormap_values(data, row)[~data.invisible.values]
                          if isinstance(row, ColormapRowConfig)
                          else visible_values(data, quantity, unit))
                if normalization == "log" and np.any(values <= 0):
                    raise ValueError(f"{run.entry.run_name}, row {row.label}: logarithmic values must be positive")
                ranges[index, item] = extend_range(ranges[index, item], values)
        del data
    resolved: list[RowConfig] = []
    for index, row in enumerate(settings.rows):
        if isinstance(row, ColormapRowConfig):
            bounds = ranges[index, 0]
            if bounds is None:
                raise ValueError(f"No visible values for row {row.label}")
            low = bounds[0] if row.style.vmin is None else real_scalar(row.style.vmin)
            high = bounds[1] if row.style.vmax is None else real_scalar(row.style.vmax)
            if not low < high or low > bounds[0] or high < bounds[1]:
                raise ValueError(f"Row {row.label}: invalid, degenerate or excluding color bounds; visible range={bounds}")
            style = replace(row.style, vmin=low, vmax=high)
            if style.norm.type == "centered" and not low < real_scalar(style.norm.vcenter) < high:
                raise ValueError(f"Row {row.label}: set explicit color bounds straddling the configured center")
            style.resolve_norm()
            resolved.append(replace(row, style=style))
        else:
            components = []
            for item, component in enumerate(row.components):
                bounds = ranges[index, item]
                if bounds is None:
                    raise ValueError(f"No visible values for contour {component.quantity}")
                levels = component.style.levels.levels(np.asarray(bounds, dtype=np.float64))
                values = [real_scalar(value) for value in np.asarray(levels).ravel()]
                if not values:
                    raise ValueError(f"No global contour levels for {component.quantity}")
                style = replace(component.style, levels=replace(component.style.levels, values=values))
                components.append(replace(component, style=style))
            resolved.append(replace(row, components=tuple(components)))
    return replace(settings, rows=tuple(resolved))


def validate_resolved_run(data: ChevalierPlotData, settings: SweepSettings, name: str) -> None:
    """Detect data that no longer fit resolved global color bounds"""
    for row in settings.rows:
        if isinstance(row, ColormapRowConfig):
            values = colormap_values(data, row)[~data.invisible.values]
            low, high = row.style.vmin, row.style.vmax
            if low is None or high is None:
                raise ValueError("Resolve global row bounds before drawing")
            if np.any((values < low) | (values > high)):
                raise ValueError(f"{name}, row {row.label}: data exceed resolved global color bounds")


def draw_matrix_page(runs: tuple[SelectedRun, ...], settings: SweepSettings) -> Figure:
    """Draw one page by loading one RUN at a time and sharing colorbars by row"""
    if not runs:
        raise ValueError("A matrix page requires at least one RUN")
    rows = settings.rows
    layout = pu.PlotLayout(
        figure=pu.FigureConfig((settings.cell_size[0]*len(runs), settings.cell_size[1]*len(rows)),
                               dpi=settings.dpi, layout_engine="constrained"),
        axes=rows[0].axes,
        subplots=pu.SubplotConfig(nrows=len(rows), ncols=len(runs), sharex=True, sharey=True),
    )
    fig, axes = layout.create()
    mappables: list[QuadMesh | None] = [None] * len(rows)
    try:
        for col, run in enumerate(runs):
            data = load_run(run, settings, report_missing_hashes=False)
            validate_resolved_run(data, settings, run.entry.run_name)
            for index, row in enumerate(rows):
                ax = axes[index, col]
                show_legend = index == 0 and col == 0
                if isinstance(row, ContourRowConfig):
                    draw_contour_panel(ax, data, row, show_legend=show_legend)
                else:
                    mappables[index] = draw_colormap_panel(ax, data, row, show_legend=show_legend)
                if index == 0:
                    ax.set_title(run.title(), fontsize=settings.title_fontsize)
                if index != len(rows)-1:
                    ax.set_xlabel("")
                    ax.tick_params(axis="x", which="both", labelbottom=False)
                if col != 0:
                    ax.set_ylabel("")
                    ax.tick_params(axis="y", which="both", labelleft=False)
                else:
                    label = pu.AxesConfig._single_axis(row.axes.y).label.text
                    ax.set_ylabel(f"{row.label}\n{label}")
            del data
        for index, row in enumerate(rows):
            if isinstance(row, ColormapRowConfig):
                mesh = mappables[index]
                if mesh is None:
                    raise RuntimeError("Missing row mappable")
                pu.ColorbarComponent(mesh, row.style.colorbar).draw(fig, list(axes[index, :]))
        return fig
    except Exception:
        plt.close(fig)
        raise


def write_sweep_pdf(plan: SweepPlan, output: Path) -> None:
    """Publish every matrix page atomically without overwriting sources"""
    if output.resolve() in {path.resolve() for path in plan.protected}:
        raise ValueError("Output must differ from all manifest, input, CSV and YAML sources")
    if output.suffix.lower() != ".pdf":
        raise ValueError("Output must have a .pdf suffix")
    if not plan.runs:
        raise ValueError("No selected RUNs")
    settings = resolve_rows(plan.runs, plan.settings)
    temporary: Path | None = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=output.parent, prefix=".chevalier_sweep_", suffix=".pdf", delete=False) as file:
            temporary = Path(file.name)
        with PdfPages(temporary) as pdf:
            width = settings.columns_per_page
            for offset in range(0, len(plan.runs), width):
                fig = draw_matrix_page(plan.runs[offset:offset+width], settings)
                try:
                    pdf.savefig(fig)
                finally:
                    plt.close(fig)
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def thin_runs(entries: list[RunEntry], max_runs: int = 5) -> list[RunEntry]:
    """Select evenly spaced RUNs while preserving manifest order"""
    if max_runs < 1:
        raise ValueError("max_runs must be positive")
    if len(entries) <= max_runs:
        return entries
    indices = np.linspace(0, len(entries) - 1, max_runs, dtype=int)
    return [entries[int(index)] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sweep-index", "peak-grid-path", "model-parameter-table-path", "obslc",
                 "contour-plotconfig", "colormap-plotconfig", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--bestfit-points-path", type=Path)
    parser.add_argument("--columns-per-page", type=int)
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()
    try:
        contours = pu.PlotConfig.from_yaml(args.contour_plotconfig)
        maps = pu.PlotConfig.from_yaml(args.colormap_plotconfig)
        settings = build_plot_config(contours, maps)
        if args.columns_per_page is not None:
            settings = replace(settings, columns_per_page=args.columns_per_page)
        validate_settings(settings)
        entries = thin_runs(read_manifest(args.sweep_index), max_runs=5)
        protected = [args.sweep_index, args.obslc, args.contour_plotconfig, args.colormap_plotconfig]
        for entry in entries:
            protected.extend((entry.input_path, within_run(entry, args.peak_grid_path),
                              within_run(entry, args.model_parameter_table_path)))
            if args.bestfit_points_path is not None:
                protected.append(within_run(entry, args.bestfit_points_path))
        runs = select_runs(entries, peak_path=args.peak_grid_path, model_path=args.model_parameter_table_path,
                           obslc=args.obslc, bestfit_path=args.bestfit_points_path, skip_missing=args.skip_missing)
        write_sweep_pdf(SweepPlan(runs, settings, tuple(protected)), args.output)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    pages = math.ceil(len(runs)/settings.columns_per_page)
    print(f"Saved: {args.output}; RUNs={len(runs)}; rows={len(settings.rows)}; pages={pages}", flush=True)


if __name__ == "__main__":
    main()
