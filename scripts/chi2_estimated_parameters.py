from typing import Any
import argparse
from pathlib import Path
from module.dataframe_processors import filter_df_value_window
from module.utilities import filereaders as fr
from module.utilities import filewriters as fw
from module.strenums import KeyNames, EstimationConfigNames
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from module import dataframe_utils

def filter_df_obs_time_window(df_obs:pd.DataFrame,conf_filter:dict[str,Any]):
    return filter_df_value_window(
        df_obs,
        column_name = KeyNames.T,
        min = conf_filter[EstimationConfigNames.MIN],
        max = conf_filter[EstimationConfigNames.MAX]
    )

def calculate_chi2(
    y_model:NDArray[np.float64],
    y_obs:NDArray[np.float64],
    y_err:NDArray[np.float64],
):
    chi_arr = (y_model - y_obs)/y_err
    return np.sum(chi_arr**2).item()


parser = argparse.ArgumentParser()

parser.add_argument(
    "chi2_table",
    type=Path,
)
parser.add_argument(
    "--parameter_table",
    type=Path,
    required=True
)
parser.add_argument(
    "--output",
    type=Path,
    required=True
)
args = parser.parse_args()

chi2_path:Path = args.chi2_table
df_chi2 = fr.read_csv(chi2_path)
metadata_chi2 = fr.read_keyvalue(chi2_path)

param_table_path:Path = args.parameter_table
metadata_param_table = fr.read_keyvalue(param_table_path)
df_param_table_raw = fr.read_csv(param_table_path)
df_param_table = df_param_table_raw.set_index("idx")

outpath:Path = args.output
outpath.parent.mkdir(parents=True,exist_ok=True)

df_set = dataframe_utils.build_dfs_grouped(
    df_long = df_chi2,
    group_by = KeyNames.NU
)

rows:list[pd.Series] = []
for (nu, df_nu) in df_set:
    df_nu_setidx = df_nu.set_index("idx")

    idx_chi2min = df_nu_setidx["chi2"].idxmin()

    row_chi2min = df_nu_setidx.loc[idx_chi2min]
    row_param = pd.Series(df_param_table.loc[idx_chi2min])
    row_estimated = pd.concat([row_chi2min,row_param])
    
    rows.append(row_estimated)

df_output = pd.DataFrame(rows)

fw.write_csv_with_params(
    df_output,
    {
        **metadata_param_table,
        **metadata_chi2
    },
    outpath
)
