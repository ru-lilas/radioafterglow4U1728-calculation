from typing import Any, cast
import argparse
from pathlib import Path
from module.dataframe_processors import filter_df_value_window
from module.utilities import filereaders as fr
from module.utilities import filewriters as fw
from module.strenums import KeyNames, EstimationConfigNames, Chi2Columns
import pandas as pd
from module import dataframe_utils
from module import mystatistics

def filter_df_obs_time_window(df_obs:pd.DataFrame,conf_filter:dict[str,Any]):
    return filter_df_value_window(
        df_obs,
        column_name = KeyNames.T,
        min = conf_filter[EstimationConfigNames.MIN],
        max = conf_filter[EstimationConfigNames.MAX]
    )

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

outpath:Path = args.output
outpath.parent.mkdir(parents=True,exist_ok=True)

df_raw = fr.read_csv(inpath)
metadata = fr.read_keyvalue(inpath)
df_nu = dataframe_utils.build_dfs_grouped(
    df_long = df_raw,
    group_by = KeyNames.NU
)

rows:list[pd.Series] = []
for (nu, df_nu) in df_nu:
    idx_chi2min = df_nu["reduced_chi2"].idxmin()
    row_estimated = pd.Series(df_nu.loc[idx_chi2min])
    chi2_min = cast(float,row_estimated[Chi2Columns.CHI2])
    ndof = cast(int,row_estimated[Chi2Columns.NDOF])
    conf = mystatistics.load_chi2_test_conf()
    result_chi2test = conf.test(chi2_min,ndof)
    row_estimated[Chi2Columns.P_VALUE] = result_chi2test.p_value
    row_estimated[Chi2Columns.SIGMA] = result_chi2test.significance
    row_estimated[Chi2Columns.REJECT] = result_chi2test.reject
    rows.append(row_estimated)

df_output = pd.DataFrame(rows)

fw.write_csv_with_params(
    df_output,
    {
        **metadata
    },
    outpath
)
