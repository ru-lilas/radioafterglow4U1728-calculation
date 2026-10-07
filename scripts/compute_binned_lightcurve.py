"""
    compute_binned_lightcurve.py
    continuous lightcurveに対してbin平均をとる
"""

import argparse
import pandas as pd
from pathlib import Path

from module import mystatistics
from module import utils
from module.inputs_as_dataclass import GeneralInputs
from module.mydataclasses import LongformatDataFrame, QuantityArray1D, QuantityData, WideFormatDataFrames
from module.mystatistics import CalculateChiSquared
from module.observation import LongformatLightcurve
from module.utils import FileReader, FileWriter
from module.synchrotron_numerical_table import DimensionlessThermalLuminosityGrid, ThermalSynchrotronIntegrals
from module.models import ColumnNames, DefaultUnits, MetadataNames, ModelParameters, SynchrotronResults
from module.calculation_lightcurve import (
    ChiSquaredResult,
    ContinuousLightcurve,
    )
from module.dataframe_utils import groupby_to_dict

def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--continuous_lightcurves",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--obslc",
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
    conf_chi2 = GeneralInputs.from_yaml(args.input).chi_squared

    longdf_contiuous = FileReader.table_from_csv(args.continuous_lightcurves)
    nu_column = ColumnNames.NU
    dfgroup_nu = groupby_to_dict(
        longdf_contiuous,ColumnNames.NU
    )

    obslc_long = LongformatLightcurve.from_csv(args.obslc)
    results = []
    nu_unit = DefaultUnits.nu
    for nu_value,df_nu in dfgroup_nu.items():
        if not isinstance(nu_value,float):
            raise TypeError(
                f"nu_valueのtypeが不正です:{type(nu_value)}"
            )
        nu = QuantityData(
            value=df_nu.attrs[ColumnNames.NU],
            unit=df_nu.attrs[MetadataNames.NU_UNIT]
        )
        obs_lc = obslc_long.select_frequency(nu,unit=nu.unit)
        obs_lc_sampled = obs_lc.select_timewindow(conf_chi2.timewindow)
        binning = obs_lc_sampled.time_bin_bounds(conf_chi2.timewindow)

        dfgroup_model = groupby_to_dict(df_nu,ColumnNames.MODEL_INDEX)

        for model_index,df_cand in dfgroup_model.items():
            if not isinstance(model_index,int):
                raise TypeError(
                    f"model_indexのtypeが不正です:{type(model_index)}"
                )
            lc = ContinuousLightcurve.from_df(
                model_index,
                nu,
                df_cand,
            )
            lc_with_zero = lc.insert_zero_boundary()
            lc_binned = lc_with_zero.bin_average(
                binning,
                drop_incomplete_bin=True
            )
            results.append(lc_binned.to_df(
                fnu_bg=obs_lc_sampled.fnu_background
            ))
    longdf_result = pd.concat(
        results,
        axis="index"
    )
    FileWriter.df_to_csv(
        args.output,
        longdf_result,
        save_attrs=True
    )
    return

if __name__ == "__main__":
    main()
