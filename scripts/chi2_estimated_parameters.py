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
    "chi2_colormap",
    type=Path,
)
parser.add_argument(
    "--output",
    type=Path,
    required=True
)
args = parser.parse_args()

inpath:Path = args.chi2_colormap
df = fr.read_csv(inpath)
metadata = fr.read_keyvalue(inpath)

outpath:Path = args.output
outpath.parent.mkdir(parents=True,exist_ok=True)

df_set = dataframe_utils.build_dfs_grouped(
    df_long = df,
    group_by = KeyNames.NU
)

rows:list[pd.Series] = []
for (nu, df_nu) in df_set:
    idx_chi2min = df_nu["reduced_chi2"].idxmin()
    row_estimated = df_nu.loc[idx_chi2min]
    
    rows.append(row_estimated)

df_output = pd.DataFrame(rows)

fw.write_csv_with_params(
    df_output,
    {
        **metadata
    },
    outpath
)
