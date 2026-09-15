from module import models

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from module import plot_utils
from module.utils import FileWriter
import numpy as np

from dataclasses import dataclass
from module.mydataclasses import YAMLReadable

@dataclass(frozen=True, slots=True)
class PlotConfig(YAMLReadable):
    layout: plot_utils.PlotLayoutConfig
    style: plot_utils.LineStyle

def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser()

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
    plotconf = PlotConfig.from_yaml(args.plotconfig)
    ranges = plotconf.layout.axes.range()
    theta = models.ElectronTemperature(
        values = np.logspace(
            start=np.log10(ranges.x.min),
            stop=np.log10(ranges.x.max),
            num=512
        )
    )
    f_factor = models.MaxwellJuttnerFactor()

    ranges = plotconf.layout.axes.range()
    with PdfPages(outpath) as pdf:
        fig,ax = plot_utils.create_configured_axes(plotconf.layout)
        fig.set_layout_engine("constrained")

        f_factor.plot(theta,ax,plotconf.style)

        pdf.savefig(fig)
        plt.close(fig)

if __name__ == "__main__":
    main()
