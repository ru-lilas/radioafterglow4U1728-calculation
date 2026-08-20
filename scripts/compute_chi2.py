from functools import cached_property
from typing import Any, cast
import argparse
from pathlib import Path
from math import isclose
from dataclasses import dataclass

from module.utilities import filereaders as fr
from dacite import from_dict
from module import compute_lightcurve, input_dataclasses, mydataclasses
import numpy as np
import pandas as pd

from tqdm import tqdm
from module.dataframe_processors import filter_df_value_window
from module.utilities import filewriters as fw
from module.strenums import KeyNames, EstimationConfigNames, LightcurveColumns
from module.strenums import Chi2Columns
from module.fetch_numerical_table import fetch_numerical_table_path
from module.utilities.quantity_data import QuantityData
from module import compute_lightcurve, dataframe_utils, input_reader
from module import mystatistics
from module import observation
from functools import cached_property
import astropy.units as u
from module.parameter_table import GeneralInputs
from module import chi2_fitting

parser = argparse.ArgumentParser()

parser.add_argument(
    "--parameter_table",
    type=Path,
    required=True
)
parser.add_argument(
    "--integral_table",
    type=Path,
    required=True
)
parser.add_argument(
    "--obs_lc",
    type=Path,
    required=True
)
parser.add_argument(
    "--config",
    type=Path,
    required=True
)
parser.add_argument(
    "--output",
    type=Path,
    required=True
)
args = parser.parse_args()

path_conf:Path = args.config
path_param_table:Path = args.parameter_table
path_integral_table:Path = args.integral_table
path_obslc:Path = args.obs_lc
outpath:Path = args.output
outpath.parent.mkdir(parents=True,exist_ok=True)

conf = GeneralInputs.from_yaml(path_conf)
conf_fitting = conf.chi2fitting
n_parameter = len(conf_fitting.free_parameters)
conf_sampling = conf_fitting.sampling

obslc_general = observation.LongformatLightcurve.from_csv(path_obslc)
obslc = obslc_general.extract_lightcurve(conf_sampling.nu.value)
print(f"Frequency: {conf_sampling.nu.value} {conf_sampling.nu.unit}")

obslc_selected = obslc.select_timewindow(conf_sampling.timewindow)
n_data = len(obslc_selected.df)
timewindow = obslc_selected.time_bin_bounds(conf_sampling.timewindow)

# calculate model lightcurves
lc_conf = conf_fitting.model
table_integral = compute_lightcurve.ThermalSynchrotronTable.from_csv(path_integral_table)
lc_model = compute_lightcurve.ThermalSynchrotron(table_integral)

chi2_records: list[dict[str, int | float]] = []
for lc_input in compute_lightcurve.build_inputs(path_param_table):
    lc = compute_lightcurve.compute(
        config=lc_conf,
        model=lc_model,
        input=lc_input
    )
    lc_binned = lc.bin_average(
        binning=timewindow,
        drop_incomplete_bin=True
    )
    chi2 = chi2_fitting.calculate_lightcurve_chi2(
        model= lc_binned,
        observed=obslc_selected
    )
    chi2_fitting.append_chi2_for_input(
        chi2_records,
        lc_input.values.idx,
        chi2
    )
df = pd.DataFrame(chi2_records)
df["idx"] = df["idx"].astype(int)
df = df.set_index("idx")
chi2_min_data = chi2_fitting.build_minimum_chi2_summary(
    df_chi2= df,
    n_data = n_data,
    n_param= n_parameter
)
chi2_min_data.update_df_attrs(df)
print(df.attrs)
print(df)
