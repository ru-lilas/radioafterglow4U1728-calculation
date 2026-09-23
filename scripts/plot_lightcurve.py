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
from module.compute_lightcurve import BinnedBestfitLightcurve, CalculationLightcurve, CandidateModelFitResults,PlotConfig
from module.utils import FileReader

def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--candidate_binned_lightcurves",
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

    conf = GeneralInputs.from_yaml(args.input)
    conf_plot = PlotConfig.from_yaml(args.plotconfig)
    obslc_long = LongformatLightcurve.from_csv(args.obslc)

    cand_lcs = CandidateModelFitResults.from_df_long(
        FileReader.table_from_csv(args.candidate_binned_lightcurves)
    )

    with PdfPages(outpath) as pdf:
        plotter = plot_utils.Plotter.from_config(conf_plot.layout)

        t_unit,fnu_unit = conf_plot.units

        lc = cand_lcs.models[0].lightcurve
        lc.plot(
            plotter,
            style=conf_plot.styles.model_binned,
            t_unit=t_unit,
            fnu_unit=fnu_unit
        )
        obslc = obslc_long.select_frequency(
            nu = lc.nu,
            unit = lc.nu.unit
        )
        obslc.plot(
            plotter,
            style=conf_plot.styles.observation,
            t_unit = t_unit,
            fnu_unit = fnu_unit
        )
        obslc.plot_persistent(
            plotter,
            style=conf_plot.styles.persistent,
            t_unit = t_unit,
            fnu_unit = fnu_unit
        )
        cand_lcs.metadata.plot(
            plotter,
            style=conf_plot.metadata
        )
        # metadata_string = conf_plot.make_metadata_strings(lc.params)
        # lc.plot_metadata(fig,", ".join(metadata_string))
        # lc_raw.plot(
        #     ax,
        #     style=conf_plot.styles.model,
        #     fnu_bg=fnu_bg,
        #     t_unit = t_unit,
        #     fnu_unit = fnu_unit
        # )
        pdf.savefig(plotter.fig)
        plt.close(plotter.fig)
    #
    return

if __name__ == "__main__":
    main()
