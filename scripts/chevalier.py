import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from module import plot_utils
from module.chevalier import (
    ChevalierGrid,
    PlotConfig
)
from module.inputs_as_dataclass import GeneralInputs, ModelParameterTable
from module.mydataclasses import QuantityData
from module.observation import LongformatLightcurve
from module.utils import FileReader, FileWriter

def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_table",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--plotconfig",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--obslc",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True
    )
    return parser.parse_args()

def main():
    args = parse_args()
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    df_model = FileReader.table_from_csv(args.model_table,idx="index")
    grids = ChevalierGrid.from_df(df_model)
    grids = ChevalierGrid.from_df(args.model_table)
    df_obs_long = LongformatLightcurve.from_csv(args.obslc)

    freqs = [5.5,9.0]
    nu_unit = "GHz"
    dfs_obs_nu = [
        df_obs_long.extract_lightcurve(
            QuantityData(
                value = nu,
                unit = nu_unit
            )
        )
        for nu in freqs
    ]
    obs_ranges = [
        df_obs_nu.peak
        for df_obs_nu in dfs_obs_nu
    ]

    plotconf = PlotConfig.from_yaml(args.plotconfig)
    phi_unit = plotconf.layout.axes.x.label.unit
    fnu_unit=plotconf.layout.axes.y.label.unit

    ranges = plotconf.layout.axes.range()
    with PdfPages(outpath) as pdf:
        fig,ax = plot_utils.create_configured_axes(plotconf.layout)
        fig.set_layout_engine("constrained")


        if (phi_unit is None):
            raise ValueError("phi_unitの単位が指定されていません")
        if (fnu_unit is None):
            raise ValueError("fnu_unitの単位が指定されていません")
    #
        grids.plot_main_contour(
            ax,
            plotconf=plotconf,
            phi_unit=phi_unit,
            fnu_unit=fnu_unit,
            plotranges=ranges
        )
        grids.metadata.plot(
            ax=ax,
            conf=plotconf.metadata
        )
        style_obs_ranges = plotconf.styles.obs_ranges.to_stylelist()
        for i,obs_range in enumerate(obs_ranges):
            obs_range.plot_range(
                ax,
                style=style_obs_ranges[i],
                phi_unit=phi_unit,
                fnu_unit=fnu_unit
            )
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
        ax.legend(**plotconf.legend.to_kwargs())
        pdf.savefig(fig)
        plt.close(fig)

if __name__ == "__main__":
    main()
