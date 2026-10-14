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
from module.observation import Binning, LongformatLightcurve
from module.utils import FileReader, FileWriter
from module.synchrotron_numerical_table import DimensionlessThermalLuminosityGrid, ThermalSynchrotronIntegrals
from module.models import ColumnNames, DefaultUnits, MetadataNames, ModelParameters, SynchrotronResults
from module.calculation_lightcurve import (
    ChiSquaredResult,
    ContinuousLightcurve,
    )
from module.dataframe_utils import groupby_to_dict
from module.timing_utils import timed

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "input",
        "continuous_lightcurves",
        "binning",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args()

@timed
def main():
    args = parse_args()

    longdf_contiuous = FileReader.table_from_csv(args.continuous_lightcurves)
    nu_column = ColumnNames.NU
    dfgroup_nu = groupby_to_dict(
        longdf_contiuous,ColumnNames.NU
    )
    df_binning = FileReader.table_from_csv(args.binning)
    dfgroup_binning = groupby_to_dict(
        df_binning,ColumnNames.NU
    )

    results = []
    nu_unit = DefaultUnits.nu
    for nu_value,df_nu in dfgroup_nu.items():
        if not isinstance(nu_value,float):
            raise TypeError(
                f"nu_valueのtypeが不正です:{type(nu_value)}"
            )
        nu = QuantityData(
            value=nu_value,
            unit=df_nu.attrs[MetadataNames.NU_UNIT]
        )
        binning = Binning.from_df(dfgroup_binning[nu_value])

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
            df = lc_binned.to_df(
                nu_unit=nu.unit,
            )
            results.append(df)
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
