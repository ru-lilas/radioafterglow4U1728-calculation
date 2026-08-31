from module import dataframe_processors, dataframe_utils
from module.parameter_table import GeneralInputs
from module.utilities import filereaders as fr
from pathlib import Path
import argparse
from module.plot import curve
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from module.compute_lightcurve import BinnedCalculationLightcurve, BinnedPredictedLightcurve, CalculationLightcurve, PlotConfig
from module import plot_utils,observation

def fetch_obslc_nu(
    path:Path,
    nu:float
)->observation.Lightcurve:
    obslc_long = observation.LongformatLightcurve.from_csv(path)
    return obslc_long.extract_lightcurve(nu)

def main(args:argparse.Namespace):
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    lc = BinnedPredictedLightcurve.from_csv(
        args.lc
    )
    conf = GeneralInputs.from_yaml(args.config)
    conf_plot = PlotConfig.from_yaml(args.plotconfig)
    obslc_nu = fetch_obslc_nu(
        path=args.obslc,
        nu=conf.chi2fitting.sampling.nu.value
    )

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
        obslc_nu.plot(
            ax,
            style=conf_plot.styles.observation,
            t_unit=t_unit,
            fnu_unit=fnu_unit
        )
        obslc_nu.plot_persistent(
            ax,
            style=conf_plot.styles.persistent,
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
        "--plotconfig",
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
