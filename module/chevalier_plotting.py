"""Shared preparation and overlays for Chevalier contour and colormap plots."""
from __future__ import annotations
import argparse
from dataclasses import dataclass, fields
from pathlib import Path
import pandas as pd
from module import plot_utils, dataframe_accessors
from module.chevalier import PeakGrid, PlotRegion, RelativisticBoundary, ObservedRegion
from module.models import ColumnNames, MetadataNames, ModelParameters
from module.mydataclasses import QuantityArray1D, QuantityData, FloatGrid, BoolGrid
from module.observation import LongformatLightcurve
from module.utils import FileReader

@dataclass(frozen=True,slots=True)
class PlotMetadata:
    eps_b: QuantityData
    eps_th: QuantityData
    mu_e_mu: QuantityData
    distance: QuantityData
    p_index: QuantityData
    f_e: QuantityData

    def to_dict(self) -> dict[str, QuantityData]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "peak_grid",
        "model_parameter_table",
        "obslc",
        "plotconfig",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--chevalier_bestfit_points",
        type=Path
    )
    return parser.parse_args()


@dataclass(frozen=True, slots=True)
class ChevalierPlotData:
    config: plot_utils.PlotConfig
    model_parameters: ModelParameters
    grids: PeakGrid
    plotregion: PlotRegion
    theta: FloatGrid
    invisible: BoolGrid
    lcs_obs: dict
    df_bestfits: pd.DataFrame | None

    @classmethod
    def from_args(cls, args):
        config = plot_utils.PlotConfig.from_yaml(args.plotconfig)
        models = ModelParameters.from_df(FileReader.table_from_csv(args.model_parameter_table))
        grids = PeakGrid.from_df(FileReader.table_from_csv(args.peak_grid))
        axes = config.layout.axes
        region = grids.reduce_to_plotregion(axes.range(), phi_unit=axes.x_unit, fnu_unit=axes.y_unit)
        theta = grids.theta_grid(models).to_FloatGrid("")
        invisible = region.out_of_plotregion.take_or([theta.less_than(1.0), theta.greater_than(10.0)])
        observations = LongformatLightcurve.from_csv(args.obslc).group_by_frequency()
        bestfits = FileReader.table_from_csv(args.chevalier_bestfit_points) if args.chevalier_bestfit_points is not None else None
        return cls(config, models, grids, region, theta, invisible, observations, bestfits)

    def title_strings(self):
        model_parameters = self.model_parameters
        plotmetadataconfig = self.config.metadata
        metadata = PlotMetadata(
            eps_b=QuantityData(model_parameters.eps_b,""),
            eps_th=QuantityData(model_parameters.eps_th,""),
            f_e=QuantityData(model_parameters.powerlaw.f_e,""),
            p_index=QuantityData(model_parameters.powerlaw.p_index,""),
            mu_e_mu=QuantityData(model_parameters.molecular_weight.to_ratio(),""),
            distance=model_parameters.distance,
        )
        strings: list[str] = []
        for name, quantity in metadata.to_dict().items():
            plot_utils.MetadataComponent(
                quantity=quantity,
                config=plotmetadataconfig.items[name],
            ).add_strings(strings)

        return strings

    def draw_boundaries(self, plotter):
        components = self.config.components
        for name, critical in (("non_relativistic_boundary", 1.0), ("ultra_relativistic_boundary", 10.0)):
            boundary = RelativisticBoundary.build(
                plotregion=self.plotregion, theta=self.theta, theta_crit=critical,
                style=components[name].resolve_line_style(0),
            )
            boundary.plot(plotter)
            boundary.build_label(components[name + "_label"].text_style_config).plot(plotter)

    def draw_observations(self, plotter):
        axesconfig = self.config.layout.axes
        x_unit, y_unit = axesconfig.x_unit, axesconfig.y_unit
        plotcomponents = self.config.components
        lcs_obs, df_bestfits = self.lcs_obs, self.df_bestfits
        legends = []
        # observation
        for i,lc_obs in enumerate(lcs_obs.values()):
            obs_peak = lc_obs.peak
            obs_region = ObservedRegion.build(
                x=obs_peak.phi_peak.value_in(axesconfig.x_unit),
                y=obs_peak.fnu_peak.value_in(axesconfig.y_unit),
                x_err=obs_peak.phi_peak_err.value_in(axesconfig.x_unit),
                y_err=obs_peak.fnu_peak_err.value_in(axesconfig.y_unit),
                style=plotcomponents["obs_ranges"].resolve_rectangle_style(i)
            )
            patch = obs_region.plot(plotter)
            legends.append(
                plot_utils.LegendEntry(
                    plotcomponents["obs_ranges"],
                    patch,
                    value=obs_peak.nu.value
                )
            )
            # bestfit points
            if df_bestfits is not None:
                phi_peak:QuantityData = df_bestfits.q.array1d(
                    column=ColumnNames.PHI_PEAK,
                    unit_key=MetadataNames.PHI_UNIT
                ).index(i)
                phi_peak_err = QuantityArray1D(
                    df_bestfits[ColumnNames.PHI_PEAK_ERR].to_numpy(),
                    df_bestfits.attrs[MetadataNames.PHI_UNIT]
                ).index(i)
                fnu_peak = QuantityArray1D(
                    df_bestfits[ColumnNames.FNU_PEAK_NET].to_numpy(),
                    df_bestfits.attrs[MetadataNames.FNU_UNIT]
                ).index(i)
                plotter.errorbar(
                    x=phi_peak.value_in(x_unit),
                    y=fnu_peak.value_in(y_unit),
                    yerr=[0],
                    xerr=phi_peak_err.value_in(x_unit),
                    style=plotcomponents["bestfits"].resolve_errorbar_style(i)
                )
        plotter.legend(entries=legends, config=plotcomponents["legend"])
