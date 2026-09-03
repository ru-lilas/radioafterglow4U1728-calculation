import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from module import plot_utils
from module.chevalier import (
    ChevalierGrid,
    PlotConfig
)
from module.inputs_as_dataclass import PhysicalParameters
from module.parameter_table import LambdaPeakTable

def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--peak_table",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--plotconfig",
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

    grids = ChevalierGrid.from_yaml(args.input)
    peak_table = LambdaPeakTable.from_csv(args.peak_table)

    plotconf = PlotConfig.from_yaml(args.plotconfig)
    with PdfPages(outpath) as pdf:
        fig,ax = plot_utils.create_configured_axes(plotconf.layout)
        fig.set_layout_engine("constrained")

        phi_unit = plotconf.layout.axes.x.label.unit
        fnu_unit=plotconf.layout.axes.y.label.unit

        if (phi_unit is None):
            raise ValueError("phi_unitの単位が指定されていません")
        if (fnu_unit is None):
            raise ValueError("fnu_unitの単位が指定されていません")

        grids.plot(
            ax,
            peak_table=peak_table,
            contourconfs=plotconf.contours,
            phi_unit=phi_unit,
            fnu_unit=fnu_unit,
        )
        pdf.savefig(fig)
        plt.close(fig)

if __name__ == "__main__":
    main()
