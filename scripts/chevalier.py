import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from module import plot_utils
from module.chevalier import PeakGrid
from module.models import ModelParameters
from module.mydataclasses import QuantityData
from module.observation import LongformatLightcurve
from module.utils import FileReader, FileWriter

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

    df_peak = FileReader.table_from_csv(args.peak_grid)
    grids = PeakGrid.from_df(df_peak)

    layout = plotconfig.layout
    axesconfig = layout.axes
    plotcomponents = plotconfig.components
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

        # A_wind contour
        style = plotcomponents["a_wind"].contour_style_config
        grid = grids.a_wind_grid(model_parameters).to_FloatGrid(style.unit)
        grid_visible = grid.masking(is_out_of_plotregion)

        ctr = plotter.contour(
            x=plotregion.x_in_plotregion.values,
            y=plotregion.y_in_plotregion.values,
            z=grid_visible.values,
            style=style
        )
        plotter.contour_labels(ctr,style)

        # beta_sh contour
        style = plotcomponents["beta_sh"].contour_style_config
        grid = grids.beta_sh_grid(model_parameters).to_FloatGrid(style.unit)
        grid_visible = grid.masking(is_out_of_plotregion)

        ctr = plotter.contour(
            x=plotregion.x_in_plotregion.values,
            y=plotregion.y_in_plotregion.values,
            z=grid_visible.values,
            style=style
        )
        plotter.contour_labels(ctr,style)

        #
        #
        # grids.plot_main_contour(
        #     ax,
        #     plotconf=plotconf,
        #     phi_unit=phi_unit,
        #     fnu_unit=fnu_unit,
        #     plotranges=ranges
        # )
        # grids.metadata.plot(
        #     ax=ax,
        #     conf=plotconf.metadata
        # )
        # style_obs_ranges = plotconf.styles.obs_ranges.to_stylelist()
        # for i,obs_range in enumerate(obs_ranges):
        #     obs_range.plot_range(
        #         ax,
        #         style=style_obs_ranges[i],
        #         phi_unit=phi_unit,
        #         fnu_unit=fnu_unit
        #     )
    #     est_params = grids.select_parameters_in(
    #         obs_scat,
    #         peak_table,
    #         phi_unit=phi_unit,
    #         fnu_unit=fnu_unit
    #     )
    #     df = est_params.to_df(
    #         phi_unit=phi_unit,
    #         fnu_unit=fnu_unit,
    #         lnu_unit="erg s-1 Hz-1"
    #     )
    #
        # ax.legend(**plotconf.legend.to_kwargs())
        pdf.savefig(fig)
        plt.close(fig)

if __name__ == "__main__":
    main()
