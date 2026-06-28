import argparse
from functools import cached_property
from pathlib import Path
from typing import Any,TypeAlias
from numpy.typing import NDArray
from module.utilities import filewriters as fw
import pandas as pd
from module import dataframe_processors as dfp
from dataclasses import dataclass
import numpy as np

NDArray64:TypeAlias = NDArray[np.float64]

@dataclass
class ObservationLightcurve:
    t: NDArray64
    t_err: float
    nu: float
    fnu: NDArray64
    fnu_err: NDArray64

    @cached_property
    def fnu_bg(self):
        mask = (self.t < 0.0)
        return float(np.mean(self.fnu[mask]))

    @cached_property
    def fnu_net(self):
        return np.maximum(self.fnu - self.fnu_bg, 0.0)

    @cached_property
    def df(self):
        return pd.DataFrame({
            "nu": self.nu,
            "t": self.t,
            "t_err": self.t_err,
            "fnu": self.fnu,
            "fnu_err": self.fnu_err,
            "fnu_bg": self.fnu_bg,
            "fnu_net": self.fnu_net
        })

def read_rawdata(filepath:Path,column_names:list[str]):
    return pd.read_csv(
        filepath,
        sep=r"\s+",
        header=0,
        comment="#",
        names=column_names
    )

unit_data:dict[str,str] = {
  "t_unit": "min",
  "fnu_unit": "mJy",
  "nu_unit": "GHz",
}
t_err = 1.0
nu_value_arr:list[float] = [5.5,9.0]
column_names_input:list[str] = [
    "t",
    "fnu_5.5",
    "fnu_5.5_err",
    "fnu_9.0",
    "fnu_9.0_err"
]
column_names_output:list[str] = [
    "t",
    "t_err",
    "nu",
    "fnu",
    "fnu_err"
]

parser = argparse.ArgumentParser()
parser.add_argument(
    "input",
    type=Path,
)
parser.add_argument(
    "--output",
    type=Path,
    required=True
)
args = parser.parse_args()

outpath:Path = args.output
outpath.parent.mkdir(parents=True,exist_ok=True)

inpath:Path = args.input
df_input = read_rawdata(
    filepath=inpath,
    column_names=column_names_input
)
metadata_output:dict[str,Any] = {
    **unit_data
}

dfs:list[pd.DataFrame] = []
for nu_value in nu_value_arr:
    obs_lc_nu = ObservationLightcurve(
        t = dfp.convert_ndarray(df_input,"t"),
        t_err = t_err,
        nu = nu_value,
        fnu = dfp.convert_ndarray(df_input,f"fnu_{nu_value}"),
        fnu_err=dfp.convert_ndarray(df_input,f"fnu_{nu_value}_err")
    )
    dfs.append(obs_lc_nu.df)
df = pd.concat(dfs,ignore_index=True)

fw.write_csv_with_params(
    df,
    metadata_output,
    outpath
)
