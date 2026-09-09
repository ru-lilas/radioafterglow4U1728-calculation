from module import dataframe_processors, dataframe_utils
from module.inputs_as_dataclass import GeneralInputs
from module.mydataclasses import QuantityData
from module.utilities import filereaders as fr
from pathlib import Path
import argparse
from module.plot import curve
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from module import plot_utils,observation
from module.observation import LongformatLightcurve
from module.compute_lightcurve import BinnedBestfitLightcurve, CalculationLightcurve,PlotConfig

def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--calclc",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--calclc_raw",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--obslc",
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

    lc = BinnedBestfitLightcurve.from_csv(
        args.calclc
    )
    lc_raw = CalculationLightcurve.from_csv(args.calclc_raw)
    conf = GeneralInputs.from_yaml(args.input)
    conf_plot = PlotConfig.from_yaml(args.plotconfig)
    obslc_long = LongformatLightcurve.from_csv(args.obslc)
    obslc_nu = obslc_long.extract_lightcurve(conf.chi2fitting.sampling.nu)
    with PdfPages(outpath) as pdf:
        fig,ax = plot_utils.create_configured_axes(conf_plot.layout)
        fig.set_layout_engine("constrained")

        t_unit = conf_plot.layout.axes.x.label.unit
        fnu_unit = conf_plot.layout.axes.y.label.unit
        if (t_unit is None) or (fnu_unit is None):
            raise ValueError(
                "unitが設定されていません."
            )
        fnu_bg = QuantityData(
            value = obslc_nu.fnu_persistent.values[0],
            unit = fnu_unit
        )
        metadata_string = conf_plot.make_metadata_strings(lc.params)
        lc.plot_metadata(fig,", ".join(metadata_string))
        lc.plot_step(
            ax,
            conf_plot.styles.model_binned,
            t_unit=t_unit,
            fnu_unit=fnu_unit
        )
        lc_raw.plot(
            ax,
            fnu_bg=fnu_bg,
            t_unit = t_unit,
            fnu_unit = fnu_unit
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
    #
    return

if __name__ == "__main__":
    main()
