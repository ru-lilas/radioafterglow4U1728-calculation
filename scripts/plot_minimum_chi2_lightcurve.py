from module import dataframe_processors, dataframe_utils
from module.inputs_as_dataclass import GeneralInputs
from dataclasses import dataclass
from module.mydataclasses import QuantityData, YAMLReadable
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
from module.calculation_lightcurve import CandidateModelLightcurveContinuous, CandidateModelLightcurveTable, CandidateModelMetadata
from module import calculation_lightcurve as clc
from module.utils import FileReader

def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--minimum_chi2_lightcurve",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--candidate_models",
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
    plotconf = plot_utils.PlotConfig.from_yaml(args.plotconfig)

    # observed lightcurve
    lcs_obs = LongformatLightcurve.from_csv(args.obslc)

    # model table
    df_models = FileReader.table_from_csv(args.candidate_models)

    df_long_minchi2 = FileReader.table_from_csv(
        args.minimum_chi2_lightcurve
    )


    lc_table_minchi2 = clc.CandidateModelLightcurveTable.from_df_long(
        df_long_minchi2
    )
    lcs = [
        lc.select_unique()
        for lc in lc_table_minchi2.devide_into_frequency_groups()
    ]

    layout = plotconf.layout
    components = plotconf.components
    x_unit = layout.axes.x.label.unit
    y_unit = layout.axes.y.label.unit
    with PdfPages(outpath) as pdf:
        fig,axes = layout.create()
        for i,lc in enumerate(lcs):
            plotter = plot_utils.Plotter(axes.flat[i])
            nu = lc.nu
            
            lc_obs = lcs_obs.select_frequency(nu,unit=nu.unit)

            binned = components["binned"]
            binned_style = binned.style.resolve(i)
            plotter.step(
                edges=lc.bin_edges.to_value_in(x_unit),
                values=lc.fnu_total.to_value_in(y_unit),
                style=binned.resolve_line_style(i),
                baseline=lc.fnu_bg.value_in(y_unit)
            )
            plotter.errorbar(
                x=lc_obs.t.to_value_in(x_unit),
                y=lc_obs.fnu_total.to_value_in(y_unit),
                yerr=lc_obs.fnu_total_err.to_value_in(y_unit),
                style=components["observed"].resolve_errorbar_style(i)
            )
            plotter.line(
                x=lc_obs.t.to_value_in(x_unit),
                y=lc_obs.fnu_background.to_value_in(y_unit),
                style=components["background"].resolve_line_style(i)
            )

        pdf.savefig(fig)
        plt.close(fig)

if __name__ == "__main__":
    main()
