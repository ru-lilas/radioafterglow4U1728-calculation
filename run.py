from pathlib import Path
import argparse
from tqdm import tqdm
from module.dataframe_processors import filter_df_value_window
from module.utilities import filereaders as fr
from module.utilities import filewriters as fw
from module.strenums import KeyNames, EstimationConfigNames
import pandas as pd
from module.fetch_numerical_table import fetch_numerical_table_path
from module.utilities.quantity_data import QuantityData
import numpy as np
from numpy.typing import NDArray
from module import compute_lightcurve, dataframe_utils
from subprocess import run

parser = argparse.ArgumentParser()

parser.add_argument(
    "config",
    type=Path,
)
args = parser.parse_args()

confpath: Path = args.config
conf = fr.read_yaml(confpath)

t_min = int(conf["min"])
t_max_list = list(conf["max"])

for t_max in t_max_list:
    conf = {
        **conf,
        "max": t_max
    }
    conf_tag = f"min{t_min:02d}_max{int(t_max):02d}"
    outpath = Path(f"input/generated/chi2_sampling_{conf_tag}.yaml")
    fw.write_dict_as_yaml(conf,outpath)

    try:
        run(
            [
                "make",
                f"CONFIG_CHI2_SAMPLING={outpath}",
                "all"
            ],
            check=True
        )

    except ValueError as e:
        print(f"Skip tmax={t_max}: {e}")
        continue
