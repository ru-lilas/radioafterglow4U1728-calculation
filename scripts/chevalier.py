import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from module import plot_utils
from module.chevalier import (
    ChevalierGrid,
    ChevalierInputs,
    ChevalierGridBase,
    PlotConfig
)
from module.inputs_as_dataclass import PhysicalParameters

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

    grid = ChevalierGrid.from_yaml(args.input)
    print(grid.tau_theta)
    # grids = inputs.build_chevalier_grid(
    #     a_wind_unit="g cm-1",
    #     phi_unit="GHz min",
    #     fnu_unit="mJy"
    # )
    # plotconf = PlotConfig.from_yaml(args.plotconfig)
    # with PdfPages(outpath) as pdf:
    #     fig,ax = plot_utils.create_configured_axes(plotconf.layout)
    #     fig.set_layout_engine("constrained")
    #
    #     grids.plot(
    #         ax,
    #         contourconfs=plotconf.contours
    #     )
    #     pdf.savefig(fig)
    #     plt.close(fig)

if __name__ == "__main__":
    main()
