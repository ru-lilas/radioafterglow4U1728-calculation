"""
    take_bin_average_for_candidates.py
    candidate models lightcurveのbin平均を計算する
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

def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--candidate_lightcurves",
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
    conf_chi2 = GeneralInputs.from_yaml(args.input).chi2
    longdf_cand = LongformatDataFrame.read_from(args.candidate_lightcurves)
    dfs_cand_all = longdf_cand.parse(ColumnNames.NU)
    obslc_long = LongformatLightcurve.from_csv(args.obslc)
    results = []
    nu_unit = DefaultUnits.nu
    for nu_value,df_nu in dfs_cand_all.items():
        if not isinstance(nu_value,float):
            raise TypeError(
                f"nu_valueのtypeが不正です:{type(nu_value)}"
            )
        nu = QuantityData(
            value=nu_value,
            unit=df_nu.attrs[MetadataNames.NU_UNIT]
        )
        obs_lc = obslc_long.select_frequency(nu,unit=nu.unit)
        obs_lc_sampled = obs_lc.select_timewindow(conf_chi2.timewindow)
        binning = obs_lc_sampled.time_bin_bounds(conf_chi2.timewindow)
        candidates = LongformatDataFrame(df_nu)
        df_cands = candidates.parse(ColumnNames.MODEL_INDEX)

        for model_index,df_cand in df_cands.items():
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
            fnu_with_bg = lc_binned.fnu_with_bg(obs_lc_sampled.fnu_background)
            chi2 = ChiSquaredResult.calculate(
                model_index=model_index,
                nu_value=nu_value,
                fnu_model= fnu_with_bg,
                fnu_data = obs_lc_sampled.fnu_total,
                fnu_err  = obs_lc_sampled.fnu_total_err,
                n_params = 2
            )
            row = chi2.to_series()
            results.append(row)
    df_result = pd.DataFrame(results)
    df_result.attrs[MetadataNames.NU_UNIT] = nu_unit
    FileWriter.df_to_csv(
        args.output,
        df_result,
        save_attrs=True
    )
    return

if __name__ == "__main__":
    main()
