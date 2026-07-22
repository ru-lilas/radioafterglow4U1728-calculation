import argparse
from pathlib import Path
from module.strenums import LightcurveColumns
from module.utilities import filewriters as fw
from module.utilities import filereaders as fr
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--lc_tmp",
    type=Path,
    required=True
)
parser.add_argument(
    "--bg",
    type=Path,
    required=True
)
parser.add_argument(
    "--output",
    type=Path,
    required=True
)
args = parser.parse_args()
path_lc:Path = args.lc_tmp
path_bg:Path = args.bg
outpath:Path = args.output
outpath.parent.mkdir(parents=True,exist_ok=True)

# loading
dflong_lc = fr.read_csv(path_lc)
dflong_bg = fr.read_csv(path_bg)
metadata = fr.read_keyvalue(path_lc)

dflong_bg = dflong_bg.rename(columns={
    "mean": LightcurveColumns.BG,
    "mean_err": LightcurveColumns.BG_ERR,
})

df_output = (
    dflong_lc.merge(dflong_bg, on=LightcurveColumns.NU, how="left")
    .assign(
        fnu_net=lambda df: df[LightcurveColumns.FNU] - df[LightcurveColumns.BG],
        fnu_net_err=lambda df: np.hypot(
            df[LightcurveColumns.FNU_ERR],
            df[LightcurveColumns.BG_ERR],
        ),
    )
)

fw.write_csv_with_params(
    df_output,
    metadata,
    outpath
)
