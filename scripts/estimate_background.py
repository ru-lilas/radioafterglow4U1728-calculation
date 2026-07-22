import argparse
from dataclasses import asdict
from pathlib import Path
from module import dataframe_utils
from module import mystatistics
from module.strenums import LightcurveColumns
from module.utilities import filewriters as fw
from module.utilities import filereaders as fr
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data",
    type=Path,
    required=True
)
parser.add_argument(
    "--output",
    type=Path,
    required=True
)
args = parser.parse_args()
path_data:Path = args.data
outpath:Path = args.output
outpath.parent.mkdir(parents=True,exist_ok=True)

# loading
df_raw = fr.read_csv(path_data)
metadata = fr.read_keyvalue(path_data)

dfset = dataframe_utils.build_dfs_nu(df_raw)
rows:list[dict] = []
for nu,df_nu in dfset.items():
    df_pre_xrb = pd.DataFrame(
        df_nu.loc[df_nu[LightcurveColumns.T] < 0.0]
    )
    fnu_arr = dataframe_utils.extract_column_as_ndarray(df_pre_xrb,LightcurveColumns.FNU)
    fnu_err_arr = dataframe_utils.extract_column_as_ndarray(df_pre_xrb,LightcurveColumns.FNU_ERR)
    est_bg_nu = mystatistics.estimate_background(
        fnu=fnu_arr,fnu_err=fnu_err_arr
    )
    rows.append({
        LightcurveColumns.NU:nu,
        **asdict(est_bg_nu)
    })

# dump
df_output = pd.DataFrame(rows)
fw.write_csv_with_params(
    df_output,
    metadata,
    outpath
)
