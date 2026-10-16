"""Reusable data preparation and figure construction for Chevalier plots"""
from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
from numpy.typing import NDArray

from module import plot_utils
from module.chevalier import BoundaryLabel, ChevalierContours, PeakGrid, PlotRegion, RelativisticBoundary, ObservedRegion
from module.models import ColumnNames, MetadataNames, ModelParameters
from module.peak_cutoff_ratio import build_ratio_quantity_grid
from module.mydataclasses import QuantityArray1D, QuantityData, QuantityGrid, FloatGrid, BoolGrid
from module.observation import LongformatLightcurve, ObservedLightcurve
from module.utils import FileReader
from module.scalar_field import ScalarField2D

ParameterName = Literal["beta_sh", "a_wind"]


@dataclass(frozen=True, slots=True)
class ChevalierInputPaths:
    peak_grid: Path
    model_parameter_table: Path
    obslc: Path
    chevalier_bestfit_points: Path | None = None

    def sources(self) -> tuple[Path, ...]:
        """Return all explicitly selected CSV sources"""
        paths = (self.peak_grid, self.model_parameter_table, self.obslc)
        if self.chevalier_bestfit_points is not None:
            return paths + (self.chevalier_bestfit_points,)
        return paths

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ChevalierInputPaths:
        """Adapt legacy command arguments to validated explicit paths"""
        def required(name: str) -> Path:
            value: object = getattr(args, name, None)
            if not isinstance(value, Path):
                raise TypeError(f"{name} must be a Path")
            return value

        bestfit: object = getattr(args, "chevalier_bestfit_points", None)
        if bestfit is not None and not isinstance(bestfit, Path):
            raise TypeError("chevalier_bestfit_points must be a Path or None")
        return cls(required("peak_grid"), required("model_parameter_table"), required("obslc"), bestfit)


@dataclass(frozen=True, slots=True)
class PlotMetadata:
    eps_b: QuantityData
    eps_th: QuantityData
    mu_e_mu: QuantityData
    distance: QuantityData
    p_index: QuantityData
    f_e: QuantityData

    def to_dict(self) -> dict[str, QuantityData]:
        """Keep each metadata quantity associated with its field name"""
        return {field.name: getattr(self, field.name) for field in fields(self)}


def parse_args() -> argparse.Namespace:
    """Retain the existing single-RUN CLI for both diagram styles"""
    parser = argparse.ArgumentParser()
    for name in ("peak_grid", "model_parameter_table", "obslc", "plotconfig", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--chevalier_bestfit_points", type=Path)
    return parser.parse_args()


def shared_plane_axes(config: plot_utils.PlotConfig) -> plot_utils.AxesConfig:
    """Require one coordinate system for the shared Chevalier plot region"""
    layout = config.layout
    first = layout.axes.for_subplot(0, 0)
    for row in range(layout.subplots.nrows):
        for col in range(layout.subplots.ncols):
            current = layout.axes.for_subplot(row, col)
            if current.x != first.x or current.y != first.y:
                raise ValueError("Chevalier panels must use matching x/y units, scales and bounds")
    return first


def require_series(df: pd.DataFrame, name: str) -> pd.Series:
    """Narrow a named column to Series before processing external values"""
    if not df.columns.is_unique:
        raise ValueError("Duplicate bestfit column labels")
    if name not in df:
        raise ValueError(f"Missing bestfit column: {name}")
    column: object = df[name]
    if not isinstance(column, pd.Series):
        raise TypeError(f"Expected a single Series for {name}")
    return column


def numeric_values(df: pd.DataFrame, name: str) -> NDArray[np.float64]:
    """Validate finite real column values before converting to float64"""
    numbers: list[float] = []
    for raw in require_series(df, name).tolist():
        value: object = raw
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise TypeError(f"{name} must contain real numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain finite values")
        numbers.append(number)
    return np.asarray(numbers, dtype=np.float64)


def metadata_unit(df: pd.DataFrame, name: str) -> str:
    """Require an explicit unit string in CSV metadata"""
    unit: object = df.attrs.get(name)
    if not isinstance(unit, str) or not unit:
        raise ValueError(f"Missing or invalid unit metadata: {name}")
    return unit


def bestfit_at_frequency(df: pd.DataFrame, nu: QuantityData) -> pd.DataFrame:
    """Match bestfit points by frequency rather than CSV row order"""
    unit = metadata_unit(df, MetadataNames.NU_UNIT)
    target = nu.value_in(unit)
    matches = np.isclose(numeric_values(df, ColumnNames.NU), target, rtol=1e-10, atol=0)
    selected = df.loc[matches].copy()
    selected.attrs = df.attrs.copy()
    if len(selected) != 1:
        raise ValueError(f"Expected one bestfit row for frequency {target:g} {unit}, found {len(selected)}")
    return selected


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    path: Path
    hashes: tuple[object, ...]


def source_provenance(path: Path, df: pd.DataFrame) -> SourceProvenance:
    """Retain available fingerprints without claiming they have been verified"""
    hashes: list[object] = []
    if "config_hash" in df.attrs:
        hashes.append(df.attrs["config_hash"])
    if "config_hash" in df:
        hashes.extend(require_series(df, "config_hash").tolist())
    return SourceProvenance(path, tuple(hashes))


@dataclass(frozen=True, slots=True)
class ChevalierPlotData:
    config: plot_utils.PlotConfig
    model_parameters: ModelParameters
    grids: PeakGrid
    plotregion: PlotRegion
    theta: FloatGrid
    invisible: BoolGrid
    lcs_obs: dict[float, ObservedLightcurve]
    df_bestfits: pd.DataFrame | None
    provenance: tuple[SourceProvenance, ...] = ()
    quantity_cache: dict[str, QuantityGrid] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_paths(cls, paths: ChevalierInputPaths, config: plot_utils.PlotConfig) -> ChevalierPlotData:
        """Load one RUN using explicit CSV paths and an already loaded configuration"""
        axes = shared_plane_axes(config)
        model_table = FileReader.table_from_csv(paths.model_parameter_table)
        peak_table = FileReader.table_from_csv(paths.peak_grid)
        models = ModelParameters.from_df(model_table)
        grids = PeakGrid.from_df(peak_table)
        region = grids.reduce_to_plotregion(axes.range(), phi_unit=axes.x_unit, fnu_unit=axes.y_unit)
        theta = grids.theta_grid(models).to_FloatGrid("")
        invisible = region.out_of_plotregion.take_or([theta.less_than(1.0), theta.greater_than(10.0)])
        observations = LongformatLightcurve.from_csv(paths.obslc).group_by_frequency()
        bestfits = (FileReader.table_from_csv(paths.chevalier_bestfit_points)
                    if paths.chevalier_bestfit_points is not None else None)
        provenance = [source_provenance(paths.model_parameter_table, model_table),
                      source_provenance(paths.peak_grid, peak_table)]
        if paths.chevalier_bestfit_points is not None and bestfits is not None:
            provenance.append(source_provenance(paths.chevalier_bestfit_points, bestfits))
        return cls(config, models, grids, region, theta, invisible, observations, bestfits, tuple(provenance))

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ChevalierPlotData:
        """Retain the legacy adapter while keeping the loading API CLI-independent"""
        path: object = getattr(args, "plotconfig", None)
        if not isinstance(path, Path):
            raise TypeError("plotconfig must be a Path")
        return cls.from_paths(ChevalierInputPaths.from_args(args), plot_utils.PlotConfig.from_yaml(path))

    def title_strings(self) -> list[str]:
        """Format physical metadata through existing plot_utils components"""
        model = self.model_parameters
        metadata = PlotMetadata(
            eps_b=QuantityData(model.eps_b, ""), eps_th=QuantityData(model.eps_th, ""),
            f_e=QuantityData(model.powerlaw.f_e, ""), p_index=QuantityData(model.powerlaw.p_index, ""),
            mu_e_mu=QuantityData(model.molecular_weight.to_ratio(), ""), distance=model.distance,
        )
        strings: list[str] = []
        for name, quantity in metadata.to_dict().items():
            plot_utils.MetadataComponent(quantity, self.config.metadata.items[name]).add_strings(strings)
        return strings

    def draw_boundaries(self, plotter: plot_utils.Plotter) -> None:
        """Draw the existing temperature boundaries and their labels"""
        components = self.config.components
        for name, critical in (("non_relativistic_boundary", 1.0), ("ultra_relativistic_boundary", 10.0)):
            boundary = RelativisticBoundary.build(
                plotregion=self.plotregion, theta=self.theta, theta_crit=critical,
                style=components[name].resolve_line_style(0),
            )
            boundary.plot(plotter)
            BoundaryLabel(
                boundary.phi_boundary, boundary.fnu_boundary, boundary.theta_crit,
                components[name + "_label"].text_style_config,
            ).plot(plotter)

    def draw_observations(self, plotter: plot_utils.Plotter, *, show_legend: bool = True) -> None:
        """Draw observed peak regions and optional frequency-matched bestfit points"""
        axes = shared_plane_axes(self.config)
        components = self.config.components
        legends: list[plot_utils.LegendEntry] = []
        for index, lightcurve in enumerate(self.lcs_obs.values()):
            peak = lightcurve.peak
            region = ObservedRegion.build(
                x=peak.phi_peak.value_in(axes.x_unit), y=peak.fnu_peak.value_in(axes.y_unit),
                x_err=peak.phi_peak_err.value_in(axes.x_unit), y_err=peak.fnu_peak_err.value_in(axes.y_unit),
                style=components["obs_ranges"].resolve_rectangle_style(index),
            )
            legends.append(plot_utils.LegendEntry(components["obs_ranges"], region.plot(plotter), value=peak.nu.value))
            if self.df_bestfits is not None:
                selected = bestfit_at_frequency(self.df_bestfits, lightcurve.nu)
                phi_unit = metadata_unit(selected, MetadataNames.PHI_UNIT)
                fnu_unit = metadata_unit(selected, MetadataNames.FNU_UNIT)
                phi = QuantityArray1D(numeric_values(selected, ColumnNames.PHI_PEAK), phi_unit).index(0)
                phi_err = QuantityArray1D(numeric_values(selected, ColumnNames.PHI_PEAK_ERR), phi_unit).index(0)
                flux = QuantityArray1D(numeric_values(selected, ColumnNames.FNU_PEAK_NET), fnu_unit).index(0)
                plotter.errorbar(x=phi.value_in(axes.x_unit), y=flux.value_in(axes.y_unit), yerr=[0],
                                 xerr=phi_err.value_in(axes.x_unit), style=components["bestfits"].resolve_errorbar_style(index))
        if show_legend:
            plotter.legend(legends, components["legend"])


def page_title(data: ChevalierPlotData, page_label: str | None) -> list[str]:
    """Add an optional RUN label without replacing physical metadata"""
    strings = data.title_strings()
    if page_label is not None:
        if not isinstance(page_label, str) or not page_label.strip():
            raise ValueError("page_label must be a nonempty string or None")
        return [page_label, *strings]
    return strings


@dataclass(frozen=True, slots=True)
class ContourComponent:
    """Unresolved row configuration; rendering uses plot_utils.ContourComponent."""

    quantity: str
    style: plot_utils.ContourStyleConfig


@dataclass(frozen=True, slots=True)
class ContourRowConfig:
    label: str
    axes: plot_utils.AxesConfig
    components: tuple[ContourComponent, ...]


@dataclass(frozen=True, slots=True)
class ColormapRowConfig:
    quantity: str
    label: str
    axes: plot_utils.AxesConfig
    style: plot_utils.ColormapConfigure
    contours: tuple[ContourComponent, ...] = ()
    transform: Literal["identity", "log10"] = "identity"


RowConfig = ContourRowConfig | ColormapRowConfig


def contour_row(config: plot_utils.PlotConfig) -> ContourRowConfig:
    """Construct the legacy contour row using its existing configuration"""
    return ContourRowConfig("Contours", shared_plane_axes(config), tuple(
        ContourComponent(name, config.components[name].contour_style_config) for name in ("a_wind", "beta_sh")
    ))


def colormap_row(config: plot_utils.PlotConfig, name: str) -> ColormapRowConfig:
    """Keep a parameter selector and all its color settings in one instance"""
    style = config.components[name].colormap_style_config
    return ColormapRowConfig(name, name, shared_plane_axes(config), style)


def peak_cutoff_ratio_grid(data: ChevalierPlotData) -> QuantityGrid:
    """Calculate ratios in the exact model order used by the Chevalier coordinates"""
    peak = pd.DataFrame({ColumnNames.MODEL_INDEX: data.grids.model_indices.ravel(),
                         ColumnNames.XM: data.grids.xm.ravel()})
    return build_ratio_quantity_grid(peak, data.model_parameters, data.grids.index.values)


QUANTITY_RESOLVERS: dict[str, Callable[[ChevalierPlotData], QuantityGrid]] = {
    "a_wind": lambda data: data.grids.a_wind_grid(data.model_parameters),
    "beta_sh": lambda data: data.grids.beta_sh_grid(data.model_parameters),
    "theta": lambda data: data.grids.theta_grid(data.model_parameters),
    "xm_peak_cutoff_ratio": peak_cutoff_ratio_grid,
}


def quantity_grid(data: ChevalierPlotData, name: str) -> QuantityGrid:
    """Resolve a supported quantity without guessing a scientific definition"""
    resolver = QUANTITY_RESOLVERS.get(name)
    if resolver is None:
        raise ValueError(f"No quantity resolver for {name}; define its data contract before plotting")
    cached = data.quantity_cache.get(name)
    if cached is None:
        cached = resolver(data)
        data.quantity_cache[name] = cached
    return cached


def plane_signature(axes: plot_utils.AxesConfig) -> tuple[object, ...]:
    """Compare geometric coordinates independently of label fonts and tick styles"""
    x = plot_utils.AxesConfig._single_axis(axes.x)
    y = plot_utils.AxesConfig._single_axis(axes.y)
    return x.scale, x.limits, x.label.unit, y.scale, y.limits, y.label.unit


def parameter_values(data: ChevalierPlotData, name: str, unit: str) -> NDArray[np.float64]:
    """Validate the model-grid shape and values visible in the Chevalier plane"""
    values = quantity_grid(data, name).to_FloatGrid(unit).values
    if values.shape != data.invisible.values.shape:
        raise ValueError(f"{name} grid shape disagrees with model visibility mask")
    if values.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numeric values")
    result = np.asarray(values, dtype=np.float64).copy()
    visible = ~data.invisible.values
    if np.any(~np.isfinite(result[visible])):
        raise ValueError(f"Nonfinite visible {name} values")
    return result


def visible_values(data: ChevalierPlotData, name: str, unit: str) -> NDArray[np.float64]:
    """Return only validated visible samples for global range resolution"""
    return parameter_values(data, name, unit)[~data.invisible.values]


def colormap_values(data: ChevalierPlotData, row: ColormapRowConfig,
                    *, grid: QuantityGrid | None = None) -> NDArray[np.float64]:
    """Resolve display values without changing the physical quantity grid"""
    if grid is None:
        values = parameter_values(data, row.quantity, row.style.colorbar.label.unit)
    else:
        values = grid.to_FloatGrid(row.style.colorbar.label.unit).values.copy()
        if values.shape != data.invisible.values.shape:
            raise ValueError("Colormap grid and visibility mask shapes differ")
    visible = ~data.invisible.values
    if np.any(~np.isfinite(values[visible])):
        raise ValueError(f"Nonfinite visible {row.quantity} values")
    if row.transform == "log10":
        if np.any(values[visible] <= 0):
            raise ValueError(f"{row.quantity} must be positive before log10")
        values[visible] = np.log10(values[visible])
    elif row.transform != "identity":
        raise ValueError(f"Unsupported colormap transform: {row.transform}")
    return values


def check_panel_plane(data: ChevalierPlotData, axes: plot_utils.AxesConfig) -> None:
    """Reject a row whose coordinate units or bounds differ from its prepared data"""
    if plane_signature(axes) != plane_signature(shared_plane_axes(data.config)):
        raise ValueError("Panel coordinates disagree with the prepared Chevalier plane")


def contour_scalar_field(data: ChevalierPlotData, name: str, unit: str) -> ScalarField2D:
    """Convert a physical grid to plotting units before applying exclusions.

    Keep the original coordinate order and (ny, nx) orientation. Coordinates
    come from the full region, since masked coordinates are not valid inputs
    to ScalarField2D. The region and visibility exclusions belong on z.
    """
    values = quantity_grid(data, name).to_FloatGrid(unit).values
    if values.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numeric values")
    scalar_field = ScalarField2D(data.plotregion.x.values, data.plotregion.y.values, values)
    return scalar_field.masked(data.plotregion.out_of_plotregion.values).masked(data.invisible.values)


def local_contour_style(scalar_field: ScalarField2D,
                        style: plot_utils.ContourStyleConfig) -> plot_utils.ContourStyleConfig | None:
    """Resolve levels once, preserving explicit sweep levels and config errors.

    None means valid configuration with no locally visible contour levels.
    Automatic levels use the existing ContourLevelConfig generator. Passing
    explicit local values onward retains the low-level Plotter API.
    """
    samples = scalar_field.z.compressed()
    config = style.levels
    if config.values is not None:
        requested = np.asarray(config.values, dtype=np.float64)
    else:
        if config.scale not in ("linear", "log"):
            raise ValueError(f"Unsupported contour scale: {config.scale}")
        if isinstance(config.num, bool) or not isinstance(config.num, (int, np.integer)) or config.num < 1:
            raise ValueError("Automatic contour levels require a positive integer num")
        # Validate configuration even when the field has no visible samples.
        validation_samples = samples if samples.size else np.asarray([1.0, 10.0])
        start = config.start_value(validation_samples)
        stop = config.stop_value(validation_samples)
        if not np.isfinite(start) or not np.isfinite(stop) or start > stop:
            raise ValueError("Contour bounds must be finite and ordered")
        if config.scale == "log" and (start <= 0 or stop <= 0):
            raise ValueError("Logarithmic contour bounds must be positive")
        requested = np.asarray(config.build_levels_auto(validation_samples), dtype=np.float64)
    if requested.ndim != 1 or requested.size == 0 or not np.all(np.isfinite(requested)):
        raise ValueError("Contour levels must be a nonempty finite one-dimensional sequence")
    if samples.size == 0:
        return None
    levels = np.unique(requested)
    levels = levels[(samples.min() <= levels) & (levels <= samples.max())]
    if levels.size == 0:
        return None
    return replace(style, levels=replace(config, values=levels.tolist()))


def draw_contour_components(plotter: plot_utils.Plotter, data: ChevalierPlotData,
                            components: tuple[ContourComponent, ...]) -> None:
    """Draw only contour components and omit levels with no local crossing"""
    for component in components:
        if component.quantity in ("a_wind", "beta_sh"):
            scalar_field = contour_scalar_field(data, component.quantity, component.style.unit)
            style = local_contour_style(scalar_field, component.style)
            if style is not None:
                resolved = plot_utils.ContourComponent(
                    field=scalar_field,
                    levels=np.asarray(style.levels.values, dtype=np.float64),
                    line_style=style.lines,
                    label_style=style.label,
                    label_format=style.label_format,
                )
                resolved.draw(plotter.ax, plotter)
            continue
        values = visible_values(data, component.quantity, component.style.unit)
        if values.size == 0:
            continue
        levels = component.style.levels
        if levels.values is not None and not any(values.min() <= level <= values.max() for level in levels.values):
            continue
        ChevalierContours.build(data.plotregion, quantity_grid(data, component.quantity),
                                style=component.style, mask=data.invisible).plot(plotter)


def draw_contour_panel(ax: Axes, data: ChevalierPlotData, row: ContourRowConfig,
                       *, show_legend: bool = True) -> None:
    """Draw configured contours and overlays into an existing Axes"""
    check_panel_plane(data, row.axes)
    plot_utils.AxesConfigurator.apply(ax, row.axes)
    plotter = plot_utils.Plotter(ax)
    draw_contour_components(plotter, data, row.components)
    data.draw_boundaries(plotter)
    data.draw_observations(plotter, show_legend=show_legend)


def draw_colormap_panel(ax: Axes, data: ChevalierPlotData, row: ColormapRowConfig,
                        *, show_legend: bool = True, grid: QuantityGrid | None = None) -> QuadMesh:
    """Draw a map and overlays without creating a figure or colorbar"""
    check_panel_plane(data, row.axes)
    plot_utils.AxesConfigurator.apply(ax, row.axes)
    z = colormap_values(data, row, grid=grid)
    visible = z[~data.invisible.values]
    if np.any(~np.isfinite(visible)):
        raise ValueError(f"Nonfinite visible {row.quantity} values")
    if row.style.norm.type == "log" and np.any(visible <= 0):
        raise ValueError(f"Logarithmic {row.quantity} values must be positive")
    z[data.invisible.values] = np.nan
    plotter = plot_utils.Plotter(ax)
    mesh = plotter.colormap(data.plotregion.x.values, data.plotregion.y.values, z, row.style)
    draw_contour_components(plotter, data, row.contours)
    data.draw_boundaries(plotter)
    data.draw_observations(plotter, show_legend=show_legend)
    return mesh


def draw_contour_figure(data: ChevalierPlotData, *, page_label: str | None = None) -> Figure:
    """Build one contour figure without saving or closing a successful result"""
    subplots = data.config.layout.subplots
    if (subplots.nrows, subplots.ncols) != (1, 1):
        raise ValueError("Chevalier contours require a one-row, one-column layout")
    shared_plane_axes(data.config)
    title = page_title(data, page_label)
    fig, axes = data.config.layout.create()
    try:
        draw_contour_panel(axes[0, 0], data, contour_row(data.config))
        plotter = plot_utils.Plotter(axes[0, 0])
        metadata = title[1:] if page_label is not None else title
        lines = [", ".join(metadata[index:index + 3]) for index in range(0, len(metadata), 3)]
        if page_label is not None:
            lines.insert(0, page_label)
        plotter.title(strings=["\n".join(lines)], fontsize=data.config.metadata.fontsize)
        return fig
    except Exception:
        plt.close(fig)
        raise


def draw_parameter_map(fig: Figure, ax: Axes, data: ChevalierPlotData,
                       name: ParameterName, grid: QuantityGrid) -> QuadMesh:
    """Draw one parameter map while preserving coordinates and the existing mask"""
    row = colormap_row(data.config, name)
    mesh = draw_colormap_panel(ax, data, row, grid=grid)
    plot_utils.ColorbarComponent(mesh, row.style.colorbar).draw(fig=fig, axes=[ax])
    return mesh


def draw_colormap_figure(data: ChevalierPlotData, *, page_label: str | None = None) -> Figure:
    """Build the beta_sh and A_wind colormap figure for an externally owned PDF"""
    subplots = data.config.layout.subplots
    if (subplots.nrows, subplots.ncols) != (1, 2):
        raise ValueError("Chevalier colormap requires a one-row, two-column layout")
    shared_plane_axes(data.config)
    title = page_title(data, page_label)
    fig, axes = data.config.layout.create()
    try:
        for ax, name, grid in (
            (axes[0, 0], "beta_sh", data.grids.beta_sh_grid(data.model_parameters)),
            (axes[0, 1], "a_wind", data.grids.a_wind_grid(data.model_parameters)),
        ):
            parameter: ParameterName = "beta_sh" if name == "beta_sh" else "a_wind"
            draw_parameter_map(fig, ax, data, parameter, grid)
        text = (page_label + "\n" + ", ".join(title[1:])
                if page_label is not None else ", ".join(title))
        fig.suptitle(text, fontsize=data.config.metadata.fontsize)
        return fig
    except Exception:
        plt.close(fig)
        raise


def write_single_pdf(output: Path, build_figure: Callable[[], Figure], *, sources: Sequence[Path]) -> None:
    """Publish a single figure atomically while protecting all input files"""
    if output.resolve() in {path.resolve() for path in sources}:
        raise ValueError("Output must differ from all CSV and YAML source files")
    if output.suffix.lower() != ".pdf":
        raise ValueError("Output must have a .pdf suffix")
    fig = build_figure()
    temporary: Path | None = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=output.parent, prefix=".chevalier_", suffix=".pdf", delete=False) as file:
            temporary = Path(file.name)
        with PdfPages(temporary) as pdf:
            pdf.savefig(fig)
        temporary.replace(output)
    finally:
        plt.close(fig)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
