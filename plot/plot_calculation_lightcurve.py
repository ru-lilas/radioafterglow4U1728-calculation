from module import dataframe_processors, dataframe_utils
from module.utilities import filereaders as fr
from pathlib import Path
import argparse
from module.plot import curve
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from module.compute_lightcurve import BinnedCalculationLightcurve, CalculationLightcurve, PlotConfig
from module import plot_utils,observation

def main(args:argparse.Namespace):
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    lc = BinnedCalculationLightcurve.from_csv(
        args.lc
    )
    obslc = observation.LongformatLightcurve.from_csv(
        args.obslc
    )
    conf_plot = PlotConfig.from_yaml(args.config)

    with PdfPages(outpath) as pdf:
        fig,ax = plot_utils.create_configured_axes(conf_plot.layout)
        fig.set_layout_engine("constrained")

        t_unit = conf_plot.layout.axes.x.label.unit
        fnu_unit = conf_plot.layout.axes.y.label.unit
        if (t_unit is None) or (fnu_unit is None):
            raise ValueError(
                "unitが設定されていません."
            )

        lc.plot(
            ax,
            conf_plot.styles.model_binned,
            t_unit=t_unit,
            fnu_unit=fnu_unit
        )
        pdf.savefig(fig)
        plt.close(fig)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--lc",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--obslc",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True
    )
    
    args = parser.parse_args()
    
    main(args)
