import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from module import plot_utils
from module.chevalier import BoundaryLabel, ChevalierContours, ObservedRegion, PeakGrid, RelativisticBoundary
from module.models import ModelParameters
from module.mydataclasses import QuantityData
from module.observation import LongformatLightcurve
from module.utils import FileReader, FileWriter
from dataclasses import fields, dataclass

@dataclass(frozen=True,slots=True)
class PlotMetadata:
    eps_b: QuantityData
    eps_th: QuantityData
    mu_e_mu: QuantityData
    distance: QuantityData
    p_index: QuantityData
    eps_e: QuantityData

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
    return parser.parse_args()

def main():
    args = parse_args()
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    plotconfig = plot_utils.PlotConfig.from_yaml(args.plotconfig)

    df_model_parameter = FileReader.table_from_csv(args.model_parameter_table)
    model_parameters = ModelParameters.from_df(df_model_parameter)

    long_lcs_obs = LongformatLightcurve.from_csv(args.obslc)
    lcs_obs = long_lcs_obs.group_by_frequency()

    df_peak = FileReader.table_from_csv(args.peak_grid)
    grids = PeakGrid.from_df(df_peak)

    layout = plotconfig.layout
    axesconfig = layout.axes
    plotcomponents = plotconfig.components
    plotmetadataconfig = plotconfig.metadata
    plotrange = axesconfig.range()
    plotregion = grids.reduce_to_plotregion(
        plotrange,
        phi_unit=axesconfig.x_unit,
        fnu_unit=axesconfig.y_unit,
    )
    is_out_of_plotregion = plotregion.out_of_plotregion
    with PdfPages(outpath) as pdf:
        fig, axes = layout.create()
        plotter=plot_utils.Plotter(axes.flat[0])
        legends = []

        theta = grids.theta_grid(model_parameters).to_FloatGrid("")
        non_relativistic_boundary = RelativisticBoundary.build(
            plotregion=plotregion,
            theta=theta,
            theta_crit=1.0,
            style=plotcomponents["non_relativistic_boundary"].resolve_line_style(0)
        )
        non_relativistic_boundary_label = non_relativistic_boundary.build_label(
            plotcomponents["non_relativistic_boundary_label"].text_style_config
        )
        non_relativistic_boundary.plot(plotter)
        non_relativistic_boundary_label.plot(plotter)
        is_non_relativistic = theta.less_than(1.0)

        ultra_relativistic_boundary = RelativisticBoundary.build(
            plotregion=plotregion,
            theta=theta,
            theta_crit=10.0,
            style=plotcomponents["ultra_relativistic_boundary"].resolve_line_style(0)
        )
        ultra_relativistic_boundary_label = ultra_relativistic_boundary.build_label(
            plotcomponents["ultra_relativistic_boundary_label"].text_style_config
        )
        ultra_relativistic_boundary.plot(plotter)
        ultra_relativistic_boundary_label.plot(plotter)
        is_ultra_relativistic = theta.greater_than(10.0)

        invisible = is_out_of_plotregion.take_or([
            is_non_relativistic,
            is_ultra_relativistic
        ])

        # A_wind contour
        a_wind_contour = ChevalierContours.build(
            plotregion,
            grids.a_wind_grid(model_parameters),
            style=plotcomponents["a_wind"].contour_style_config,
            mask=invisible
        )
        a_wind_contour.plot(plotter)

        # beta_sh contour
        beta_sh_contour = ChevalierContours.build(
            plotregion,
            grids.beta_sh_grid(model_parameters),
            style=plotcomponents["beta_sh"].contour_style_config,
            mask=invisible
        )
        beta_sh_contour.plot(plotter)

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

        # metadata
        metadata = PlotMetadata(
            eps_b=QuantityData(model_parameters.eps_b,""),
            eps_th=QuantityData(model_parameters.eps_th,""),
            eps_e=QuantityData(model_parameters.powerlaw.eps_e,""),
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

        # legend
        plotter.legend(
            entries= legends,
            config = plotcomponents["legend"]
        )

        plotter.title(
            strings=strings,
            fontsize=plotmetadataconfig.fontsize,
        )

        pdf.savefig(fig)
        plt.close(fig)

if __name__ == "__main__":
    main()
