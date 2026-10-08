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
from module import models

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "plotconfig",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args()

def compute_g_factor(
    theta,
    *,
    p,
    gamma_min,
):
    g_factor = models.PowerlawDistributionFactor.compute(
        p=p,
        theta=theta,
        gamma_min=gamma_min
    )
    return g_factor

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
    theta = theta_space.geomspace()
    f_factor = models.MaxwellJuttnerFactor.compute(theta)
    g_factor = models.PowerlawDistributionFactor.compute(
        p=2.5,
        theta=theta,
        gamma_min=3.0*theta
    )

    with PdfPages(outpath) as pdf:
        fig, axes = layout.create()
        for (row,col), ax in np.ndenumerate(axes):
            axis_config = plotconf.layout.axes.for_subplot(row,col)
            plotter = plot_utils.Plotter(ax)

            plotter.line(
                x=theta,
                y=f_factor.exact,
                style=components["f_factor"].resolve_line_style(col)
            )
            plotter.line(
                x=theta,
                y=g_factor.exact,
                style=components["g_factor"].resolve_line_style(col)
            )
            plotter.line(
                x=theta,
                y=f_factor.hot,
                style=components["hot_limit"].resolve_line_style(col)
            )

        pdf.savefig(fig)
        plt.close(fig)

if __name__ == "__main__":
    main()
