import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from module import plot_utils
from module.chevalier import (
    ChevalierGrid,
    DebagPlotConfig,
    PlotConfig
)
from module.inputs_as_dataclass import GeneralInputs, ModelParameterTable
from module.mydataclasses import QuantityData
from module.observation import LongformatLightcurve
from module.parameter_table import LambdaPeakTable
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
    plotconf = DebagPlotConfig.from_yaml(args.plotconfig)
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

        grids.plot_colormaps(
            fig=fig,
            axes=axes,
            cmplots=cmplots,
            phi_unit=plotconf.layout.axes.x.label.unit,
            fnu_unit=plotconf.layout.axes.y.label.unit,
        )

        pdf.savefig(fig)
        plt.close(fig)

if __name__ == "__main__":
    main()
