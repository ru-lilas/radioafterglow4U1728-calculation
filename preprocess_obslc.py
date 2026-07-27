import argparse
from pathlib import Path
from typing import Any,cast
from numpy.typing import NDArray
from module.utilities import filewriters as fw
from module.utilities import filereaders as fr
import pandas as pd

def read_rawdata(filepath:Path,column_names:list[str]):
    return pd.read_csv(
        filepath,
        sep=r"\s+",
        header=0,
        comment="#",
        names=column_names
    )

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data",
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
path_data:Path = args.data
path_conf:Path = args.config
outpath:Path = args.output
outpath.parent.mkdir(parents=True,exist_ok=True)

# loading
conf = fr.read_yaml_pyyaml(path_conf)
df_input = read_rawdata(
    filepath=path_data,
    column_names=conf["columns"]
)

# build metadata
metadata_output:dict[str,Any] = {
    "t_bin":conf["t_bin"],
    **conf["units"]
}

# build long-format df
dfs:list[pd.DataFrame] = []
for nu_value in conf["nu_values"]:
    df_nu = cast(pd.DataFrame,df_input[["t", f"fnu_{nu_value}", f"fnu_{nu_value}_err"]])
    df_nu = df_nu.rename(columns={
        f"fnu_{nu_value}": "fnu",
        f"fnu_{nu_value}_err": "fnu_err"
    })
    df_nu["nu"] = nu_value
    print(df_nu)
    dfs.append(df_nu)
df_output = pd.concat(dfs,ignore_index=True)

fw.write_csv_with_params(
    df_output,
    metadata_output,
    outpath
)
