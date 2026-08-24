import argparse
from typing import Any
from module import dataframe_utils, fetch_numerical_table, input_reader
from pathlib import Path
import numpy as np
from module.chi2_fitting import MinimumChi2Summary
from module.utilities import filereaders as fr
from module.utilities import filewriters as fw
from module.utilities import build_nparray
import pandas as pd
from module.utilities import quantity_data
from module import dataframe_processors as dfp
from module import compute_lightcurve,observation
from module.strenums import KeyNames
import astropy.units as u
from module.parameter_table import GeneralInputs,read_as_df
from module.utils import FileReader,DataFrameUtils

parser = argparse.ArgumentParser()

parser.add_argument(
    "--estimated",
    type=Path,
    required=True
)
parser.add_argument(
    "--parameter_table",
    type=Path,
    required=True
)
# parser.add_argument(
#     "--integral_table",
#     type=Path,
#     required=True
# )
# parser.add_argument(
#     "--obs_lc",
#     type=Path,
#     required=True
# )
# parser.add_argument(
#     "--config",
#     type=Path,
#     required=True
# )
# parser.add_argument(
#     "--output",
#     type=Path,
#     required=True
# )
args = parser.parse_args()

chi2min_est = MinimumChi2Summary.from_csv(args.estimated)
idx_est:int = chi2min_est.idx

df_param = read_as_df(args.parameter_table)
row_est = DataFrameUtils.extract_row_by_index(df_param,idx_est)

print(row_est)

# conf = GeneralInputs.from_yaml(path_conf)
# conf_fitting = conf.chi2fitting
# conf_sampling = conf_fitting.sampling
#
# obslc_general = observation.LongformatLightcurve.from_csv(path_obslc)
# obslc = obslc_general.extract_lightcurve(conf_sampling.nu.value)
# obslc_selected = obslc.select_timewindow(conf_sampling.timewindow)
# timewindow = obslc_selected.time_bin_bounds(conf_sampling.timewindow)
#
# path_phys_params:Path = args.physical_parameters
# data_phys_params = input_reader.read_physical_parameters(path_phys_params)
#
# lc_conf = conf_fitting.model
# table_integral = compute_lightcurve.ThermalSynchrotronTable.from_csv(path_integral_table)
# lc_model = compute_lightcurve.ThermalSynchrotron(table_integral)

# lc = compute_lightcurve.compute(
#     config=lc_conf,
#     model=lc_model,
#     input=lc_input
# )
# lc_binned = lc.bin_average(
#     binning=timewindow,
#     drop_incomplete_bin=True
# )
# df = lc_binned.to_df(t_unit="min",fnu_unit="mJy")
