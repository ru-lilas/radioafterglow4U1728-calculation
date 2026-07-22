from typing import Any, cast
import argparse
from pathlib import Path

from tqdm import tqdm
from module.dataframe_processors import filter_df_value_window
from module.utilities import filereaders as fr
from module.utilities import filewriters as fw
from module.strenums import KeyNames, EstimationConfigNames
from module.strenums import Chi2Columns
import pandas as pd
from module.fetch_numerical_table import fetch_numerical_table_path
from module.utilities.quantity_data import QuantityData
import numpy as np
from module import compute_lightcurve, dataframe_utils, input_reader
from module import mystatistics

parser = argparse.ArgumentParser()

parser.add_argument(
    "--parameter_table",
    type=Path,
)
parser.add_argument(
    "--lightcurve_config",
    type=Path,
)
parser.add_argument(
    "--table_integral",
    type=Path,
)

args = parser.parse_args()
path_conf_lc:Path = args.lightcurve_config
path_table_integral:Path = args.table_integral
table_integral = fetch_numerical_table_path(path_table_integral)

conf_lc = input_reader.InputReader.read(path_conf_lc,input_reader.LightcurveConfigure)

nu_value:float = args.nu_value
nu_unit:str = args.nu_unit

lc = compute_lightcurve.Lightcurve(
    t = QuantityData(conf_lc.t_value_arr.nparr,conf_lc.t_unit),
    nu = QuantityData(np.asanyarray(nu_value),nu_unit),
    table_integral = table_integral
)
