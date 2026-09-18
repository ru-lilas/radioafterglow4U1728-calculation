import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from module import plot_utils
from module import chevalier
from module.chevalier import (
    ChevalierGrid,
    ColormapPlotConfig,
    PlotConfig
)
from module.inputs_as_dataclass import GeneralInputs, ModelParameterTable
from module.mydataclasses import QuantityData
from module.observation import LongformatLightcurve
from module.utils import FileWriter

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

    grids = ChevalierGrid.from_csv(args.model_table)
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

    plotconf = ColormapPlotConfig.from_yaml(args.plotconfig)
    ranges = plotconf.layout.axes.range()
    cmconf=plotconf.colormaps

    cmplots = [
        plot_utils.ColormapPlot(
            quantity=grids.theta,
            style=cmconf.theta
        ),
        plot_utils.ColormapPlot(
            quantity=grids.tau_th,
            style=cmconf.tau_th
        ),
        plot_utils.ColormapPlot(
            quantity=grids.phi_th,
            style=cmconf.phi_th
        ),
        plot_utils.ColormapPlot(
            quantity=grids.beta_sh,
            style=cmconf.beta_sh
        ),
        plot_utils.ColormapPlot(
            quantity=grids.a_wind,
            style=cmconf.a_wind
        ),
    ]

    with PdfPages(outpath) as pdf:
        fig,axes = plot_utils.create_configured_subplots(plotconf.layout)
        fig.set_layout_engine("constrained")

        phi_unit = plotconf.layout.axes.x.label.unit
        fnu_unit=plotconf.layout.axes.y.label.unit

        if (phi_unit is None):
            raise ValueError("phi_unitの単位が指定されていません")
        if (fnu_unit is None):
            raise ValueError("fnu_unitの単位が指定されていません")

        colormaps = grids.colormaps(
            cmplots=cmplots,
            phi_unit=phi_unit,
            fnu_unit=fnu_unit,
            plotranges=ranges
        )
        style_obs_ranges = plotconf.styles.obs_ranges.to_stylelist()
        for i,obs_range in enumerate(obs_ranges):
            # boolmap_in_obsrange = grids
            for (colormap,ax) in zip(colormaps,axes.flat):
                colormap.plot(fig,ax)
                obs_range.plot_range(
                    ax,
                    style=style_obs_ranges[i],
                    phi_unit=phi_unit,
                    fnu_unit=fnu_unit
                )

        pdf.savefig(fig)
        plt.close(fig)

if __name__ == "__main__":
    main()
