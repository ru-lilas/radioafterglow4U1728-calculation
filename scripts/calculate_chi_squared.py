import argparse
import pandas as pd
from pathlib import Path

from module import utils
from module.inputs_as_dataclass import GeneralInputs
from module.mydataclasses import LongformatDataFrame, QuantityArray1D, QuantityData, WideFormatDataFrames
from module.mystatistics import CalculateChiSquared
from module.observation import LongformatLightcurve
from module.utils import FileReader, FileWriter
from module.synchrotron_numerical_table import DimensionlessThermalLuminosityGrid, ThermalSynchrotronIntegrals
from module.models import ColumnNames, DefaultUnits, MetadataNames, ModelParameters, SynchrotronResults
from module.calculation_lightcurve import (
    BinnedLightcurveWithBackground,
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
        "--output",
        type=Path,
        required=True
    )
    return parser.parse_args()

def main():
    args = parse_args()
    conf_chi2 = GeneralInputs.from_yaml(args.input).chi2
    longdf_binnedlc = LongformatDataFrame.read_from(args.candidate_binned_lightcurves)
    binned_lcs = longdf_binnedlc.parse(ColumnNames.NU)
    obslc_long = LongformatLightcurve.from_csv(args.obslc)
    results = []
    nu_unit = DefaultUnits.nu
    for nu_value,longdf_nu in binned_lcs.items():
        if not isinstance(nu_value,float):
            raise TypeError(
                f"nu_valueのtypeが不正です:{type(nu_value)}"
            )
        longdf_nu.attrs[ColumnNames.NU] = nu_value
        nu = QuantityData(
            value=nu_value,
            unit=nu_unit
        )
        obs_lc = obslc_long.select_frequency(nu,unit=nu.unit)
        obs_lc_sampled = obs_lc.select_timewindow(conf_chi2.timewindow)
        binning = obs_lc_sampled.time_bin_bounds(conf_chi2.timewindow)
        lcs_nu = LongformatDataFrame(longdf_nu)
        dfs_candidate = lcs_nu.parse(ColumnNames.MODEL_INDEX)

        for model_index,df_cand in dfs_candidate.items():
            df_cand.attrs[ColumnNames.MODEL_INDEX] = model_index
            if not isinstance(model_index,int):
                raise TypeError(
                    f"model_indexのtypeが不正です:{type(model_index)}"
                )
            binned_lc = BinnedLightcurveWithBackground.from_df(df_cand)
            chi2 = ChiSquaredResult.calculate(
                model_index=model_index,
                nu_value=nu_value,
                fnu_model= binned_lc.fnu_with_bg,
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

