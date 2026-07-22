from typing import Any, cast
import argparse
from pathlib import Path

from tqdm import tqdm
from module.dataframe_processors import filter_df_value_window
from module.utilities import filereaders as fr
from module.utilities import filewriters as fw
from module.strenums import KeyNames, EstimationConfigNames, LightcurveColumns
from module.strenums import Chi2Columns
import pandas as pd
from module.fetch_numerical_table import fetch_numerical_table_path
from module.utilities.quantity_data import QuantityData
import numpy as np
from module import compute_lightcurve, dataframe_utils, input_reader
from module import mystatistics

parser = argparse.ArgumentParser()

parser.add_argument(
    "lightcurve_data",
    type=Path,
)
parser.add_argument(
    "parameter_yaml",
    type=Path,
)
parser.add_argument(
    "--integral_table",
    type=Path,
    required=True
)
parser.add_argument(
    "--observation_lc",
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

path_lightcurve:Path = args.lightcurve_data
df_long = fr.read_csv(path_lightcurve)
dfs = dataframe_utils.build_dfs_grouped(df_long,LightcurveColumns.NU)

bin_value = 60
bin_unit = "s"

for (nu,df) in dfs:
    
