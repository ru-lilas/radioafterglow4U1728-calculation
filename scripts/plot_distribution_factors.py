import argparse
from pathlib import Path
from typing import Hashable, Literal, get_args
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import fields, dataclass
from matplotlib.backends.backend_pdf import PdfPages

from module import plot_utils
from module.models import ColumnNames, FluxName, MetadataNames, ModelParameters, SwitchKeys
from module.mydataclasses import FloatVectorSpace, LongformatDataFrame, QuantityArray1D, QuantityData
from module.observation import LongformatLightcurve
from module.electron_distributions import ElectronDistribution, MaxwellJuttnerDistribution, PowerlawDistribution

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "plotconfig",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    outpath: Path = args.output
    outpath.parent.mkdir(parents=True, exist_ok=True)

    plotconf = plot_utils.PlotConfig.from_yaml(args.plotconfig)
    layout = plotconf.layout
    components = plotconf.components
    metadata_config = plotconf.metadata

    gamma_space = FloatVectorSpace(
        start=1.01e+00,
        stop =1.0e+03,
        num = 2048,
        scale="log"
    )
    f_e = 0.1
    p_index=2.5
    theta = np.array([
        5.0e-01,
        1.0e+00,
        5.0e+00
    ])
    gamma_e = gamma_space.geomspace()

    totdist = ElectronDistribution.compute(
        theta=theta,
        gamma=gamma_e,
        gamma_min=None,
        p_index=p_index,
        f_e=f_e
    )
    totfrac = totdist.integrated_number_fraction()
    metadata:dict[str,QuantityData] = {
        "f_e":QuantityData(f_e,""),
        "p_index":QuantityData(p_index,"")
    }

    with PdfPages(outpath) as pdf:
        fig, axes = layout.create()
        for (row,col), ax in np.ndenumerate(axes):
            axis_config = plotconf.layout.axes.for_subplot(row,col)
            plotter = plot_utils.Plotter(ax)

            gamma_e = totdist.gamma
            legends = []
            for i,gamma_pdf in enumerate(totdist.gamma_pdf):
                mjdist = totdist.thermal
                pldist = totdist.powerlaw
                theta = mjdist.theta
                print(f"--- theta={theta[i]:.2f} ---")
                print("f_num \t E/mc2")
                line = plotter.line(
                    x=gamma_e,
                    y=totdist.thermal.gamma_pdf[i],
                    style=components["thermal"].resolve_line_style(i)
                )
                print(
                    f"{mjdist.integrated_number_fraction()[i]:.2f}\t"
                    f"{mjdist.integrated_kinetic_energy()[i]:.2f}"
                )
                plotter.line(
                    x=gamma_e,
                    y=totdist.powerlaw.gamma_pdf[i],
                    style=components["powerlaw"].resolve_line_style(i)
                )
                print(
                    f"{pldist.integrated_number_fraction()[i]:.2f}\t"
                    f"{pldist.integrated_kinetic_energy()[i]:.2f}"
                )
                line = plotter.line(
                    x=gamma_e,
                    y=totdist.gamma_pdf[i],
                    style=components["total"].resolve_line_style(i)
                )
                print(
                    f"{totdist.integrated_number_fraction()[i]:.2f}\t"
                    f"{totdist.integrated_kinetic_energy()[i]:.2f}"
                )
                legends.append(
                    plot_utils.LegendEntry(
                        components["total"],
                        line[0],
                        value=theta[i]
                    )
                )
            plotter.legend(
                entries= legends,
                config = components["legend"]
            )
            strings: list[str] = []
            for name, quantity in metadata.items():
                plot_utils.MetadataComponent(
                    quantity=quantity,
                    config=plotconf.metadata.items[name],
                ).add_strings(strings)
            plotter.title(
                strings=strings,
                fontsize=plotconf.metadata.fontsize,
            )
        pdf.savefig(fig)
        plt.close(fig)

if __name__ == "__main__":
    main()
