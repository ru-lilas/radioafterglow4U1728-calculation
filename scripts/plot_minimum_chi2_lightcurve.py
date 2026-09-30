from module import dataframe_processors, dataframe_utils
from module.inputs_as_dataclass import GeneralInputs
from dataclasses import dataclass
from module.models import ColumnNames, FluxName, MetadataNames
from module.mydataclasses import LongformatDataFrame, QuantityArray1D, QuantityData, YAMLReadable
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
        "--minimum_chi_squared",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--candidate_binned_lightcurves",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--candidate_lightcurves",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--model_parameter_table",
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

    df_minchi2 = FileReader.table_from_csv(args.minimum_chi_squared)

    longdf_binned = LongformatDataFrame.read_from(args.candidate_binned_lightcurves)
    dfs_binned_nu = longdf_binned.parse(ColumnNames.NU)

    longdf_continuous = LongformatDataFrame.read_from(args.candidate_lightcurves)
    dfs_continuous_nu = longdf_continuous.parse(ColumnNames.NU)

    lcs_obs = LongformatLightcurve.from_csv(args.obslc)

    layout = plotconf.layout
    components = plotconf.components
    x_unit = layout.axes.x.label.unit
    y_unit = layout.axes.y.label.unit
    with PdfPages(outpath) as pdf:
        fig,axes = layout.create()
        for i,row in df_minchi2.iterrows():
            if not isinstance(i,int):
                raise TypeError(
                    f"iのtypeが不正です:{type(i)}"
                )
            minchi2 = clc.ChiSquaredResult.from_series(row)
            model_index = int(minchi2.model_index)

            plotter = plot_utils.Plotter(axes.flat[i])
            nu_value = row[ColumnNames.NU]
            if not isinstance(nu_value,float):
                raise TypeError(
                    f"nu_valueのtypeが不正です:{type(nu_value)}"
                )
            nu_unit = df_minchi2.attrs[MetadataNames.NU_UNIT]
            nu = QuantityData(
                value=nu_value,
                unit=nu_unit
            )
            lc_obs = lcs_obs.select_frequency(nu,unit=nu.unit)

            # binned lightcurve
            binned = components["binned"]
            df_binned_nu = dfs_binned_nu[nu_value]
            df_binned_plot:pd.DataFrame = df_binned_nu.loc[
                df_binned_nu[ColumnNames.MODEL_INDEX] == model_index
            ]
            df_binned_plot.attrs[ColumnNames.NU] = nu_value
            df_binned_plot.attrs[ColumnNames.MODEL_INDEX] = model_index
            binned_lc = clc.BinnedLightcurveWithBackground.from_df(df_binned_plot)
            plotter.step(
                edges=binned_lc.bin_edges.to_value_in(x_unit),
                values=binned_lc.fnu_with_bg.to_value_in(y_unit),
                style=binned.resolve_line_style(i),
                baseline=lc_obs.fnu_background.to_value_in(y_unit)[0]
            )
            
            # continuous lightcurve
            df_continuous_nu = dfs_continuous_nu[nu_value]
            df_continuous_plot:pd.DataFrame = df_continuous_nu.loc[
                df_continuous_nu[ColumnNames.MODEL_INDEX] == model_index
            ]
            print(df_continuous_plot)
            continuous_lc = clc.ContinuousLightcurve(
                model_index=model_index,
                nu=nu,
                t=QuantityArray1D.from_df(
                    df_continuous_plot,
                    ColumnNames.T,
                    unit_key=MetadataNames.T_UNIT
                ),
                fnu_net=QuantityArray1D.from_df(
                    df_continuous_plot,
                    FluxName.FNU_TOTAL,
                    unit_key=MetadataNames.FNU_UNIT
                )
            )
            plotter.line(
                x=continuous_lc.t.to_value_in(x_unit),
                y=continuous_lc.fnu_with_bg(lc_obs.fnu_background.index(0)).to_value_in(y_unit),
                style=components["continuous"].resolve_line_style(i)
            )

            plotter.line(
                x=lc_obs.t.to_value_in(x_unit),
                y=lc_obs.fnu_background.to_value_in(y_unit),
                style=components["background"].resolve_line_style(i)
            )
            plotter.errorbar(
                x=lc_obs.t.to_value_in(x_unit),
                y=lc_obs.fnu_total.to_value_in(y_unit),
                yerr=lc_obs.fnu_total_err.to_value_in(y_unit),
                style=components["observed"].resolve_errorbar_style(i)
            )
        pdf.savefig(fig)
        plt.close(fig)

if __name__ == "__main__":
    main()
