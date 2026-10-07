"""
    build_binning.py
    binを作成する
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "input",
        "obslc",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    obs = LongformatLightcurve.from_csv(args.obslc)
    group_obs = obs.group_by_frequency()
    conf = GeneralInputs.from_yaml(args.input)
    conf_timewindow = conf.chi_squared.timewindow

    results = []
    for nu_value,obslc in group_obs.items():
        binning = obslc.time_bin_bounds(conf_timewindow)
        df = binning.to_df()
        df[ColumnNames.NU] = nu_value
        results.append(df)
    longdf_result = pd.concat(
        results,
        axis="index"
    )
    longdf_result.attrs[MetadataNames.NU_UNIT] = obs.metadata.nu_unit
    FileWriter.df_to_csv(
        args.output,
        longdf_result,
        save_attrs=True
    )
    return
if __name__ == "__main__":
    main()
