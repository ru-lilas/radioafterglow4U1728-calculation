"""
    take_bin_average_for_candidates.py
    candidate models lightcurveのbin平均を計算する
"""

import argparse
import pandas as pd
from pathlib import Path

from module import mystatistics
from module.inputs_as_dataclass import GeneralInputs
from module.mydataclasses import QuantityArray1D, QuantityData, WideFormatDataFrames
from module.mystatistics import CalculateChiSquared
from module.observation import LongformatLightcurve
from module.utils import FileReader, FileWriter
from module.synchrotron_numerical_table import DimensionlessThermalLuminosityGrid, ThermalSynchrotronIntegrals
from module.models import ColumnNames, MetadataNames, ModelParameters, SynchrotronResults
from module.calculation_lightcurve import (
    BinnedLightcurveResult,
    ChiSquared,
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
    longdf_cand = FileReader.table_from_csv(args.candidate_lightcurves)
    dfs_cand_all = {
        nu: df_cand.drop(columns=ColumnNames.NU)
        for nu,df_cand in longdf_cand.groupby(ColumnNames.NU)
    }
    obslc_long = LongformatLightcurve.from_csv(args.obslc)
    dfs:list[pd.DataFrame] = []
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
        df_cands = {
            model_index: df_cand.drop(columns=ColumnNames.MODEL_INDEX)
            for model_index,df_cand in df_nu.groupby(ColumnNames.MODEL_INDEX)
        }

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
            chi2 = ChiSquared.calculate(
                model_index=model_index,
                fnu_model= fnu_with_bg,
                fnu_data = obs_lc_sampled.fnu_total,
                fnu_err  = obs_lc_sampled.fnu_total_err,
                n_params = 2
            )
            result = BinnedLightcurveResult(
                model_index=model_index,
                nu=nu,
                t_center=lc_binned.t_center,
                t_right=lc_binned.t_right,
                t_left=lc_binned.t_left,
                fnu_with_bg=fnu_with_bg,
                chi_squared=chi2.chi_squared,
                ndof=chi2.ndof
            )
            dfs.append(result.to_df())
    longdf_lc = pd.concat(
        dfs,
        axis="index"
    )
    FileWriter.df_to_csv(
        args.output,
        longdf_lc,
        save_attrs=True
    )
    return

if __name__ == "__main__":
    main()
