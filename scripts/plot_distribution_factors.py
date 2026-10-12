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
from module.electron_distributions import MaxwellJuttnerDistribution

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

    theta_space = FloatVectorSpace(
        start=1.0e-01,
        stop =1.0e+02,
        num = 256,
        scale="log"
    )
    gamma_space = FloatVectorSpace(
        start=1.0e+00,
        stop =1.0e+03,
        num = 256,
        scale="log"
    )
    f_e = 0.1
    theta = theta_space.geomspace()
    theta = np.array([
        5.0e-01,
        1.0e+00,
        5.0e+00
    ])
    gamma_e = gamma_space.geomspace()

    mjdist = MaxwellJuttnerDistribution.compute(
        f_e=f_e,
        theta=theta,
        gamma=gamma_e
    )

    with PdfPages(outpath) as pdf:
        fig, axes = layout.create()
        for (row,col), ax in np.ndenumerate(axes):
            axis_config = plotconf.layout.axes.for_subplot(row,col)
            plotter = plot_utils.Plotter(ax)

            gamma_e = mjdist.gamma
            for i,gamma_pdf in enumerate(mjdist.gamma_pdf):
                plotter.line(
                    x=gamma_e,
                    y=gamma_pdf,
                    style=components["thermal"].resolve_line_style(i)
                )

        pdf.savefig(fig)
        plt.close(fig)

if __name__ == "__main__":
    main()
