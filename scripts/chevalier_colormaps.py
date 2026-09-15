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
    with PdfPages(outpath) as pdf:
        fig,axes = plot_utils.create_configured_subplots(plotconf.layout)
        fig.set_layout_engine("constrained")

        phi_unit = plotconf.layout.axes.x.label.unit
        fnu_unit=plotconf.layout.axes.y.label.unit

        if (phi_unit is None):
            raise ValueError("phi_unitの単位が指定されていません")
        if (fnu_unit is None):
            raise ValueError("fnu_unitの単位が指定されていません")

        mesh = grids.plot_tau_colormap(
            axes[0,0],
            style=plotconf.colormaps.tau_th,
            phi_unit=phi_unit,
            fnu_unit=fnu_unit
        )
        grids.show_colorbar(
            mesh=mesh,
            fig=fig,
            ax=axes[0,0],
            style=plotconf.colormaps.tau_th
        )
        mesh = grids.plot_theta_colormap(
            axes[0,1],
            style=plotconf.colormaps.theta,
            phi_unit=phi_unit,
            fnu_unit=fnu_unit
        )
        grids.show_colorbar(
            mesh=mesh,
            fig=fig,
            ax=axes[0,1],
            style=plotconf.colormaps.theta,
        )
        mesh = grids.plot_phi_th_colormap(
            axes[1,0],
            style=plotconf.colormaps.phi_th,
            phi_unit=phi_unit,
            fnu_unit=fnu_unit
        )
        grids.show_colorbar(
            mesh=mesh,
            fig=fig,
            ax=axes[1,0],
            style=plotconf.colormaps.phi_th,
        )
        mesh = grids.plot_lnu_th_colormap(
            axes[1,1],
            style=plotconf.colormaps.lnu_th,
            phi_unit=phi_unit,
            fnu_unit=fnu_unit,
            lnu_unit=grids.metadata.lnu_unit
        )
        grids.show_colorbar(
            mesh=mesh,
            fig=fig,
            ax=axes[1,1],
            style=plotconf.colormaps.lnu_th,
        )
        pdf.savefig(fig)
        plt.close(fig)

if __name__ == "__main__":
    main()
