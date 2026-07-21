from module import dataframe_utils
from module.strenums import Chi2Columns, KeyNames, SamplingConfigNames
from module.utilities import filereaders as fr
from module.utilities import filewriters as fw
from pathlib import Path
import argparse
import pandas as pd
from module.strenums import KeyNames
from module import find_subdirs
from module.input_reader import InputReader
from module.input_dataclasses import SAMPLING
from module.strenums import FileNames,FileExtension

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data",
    type=Path,
    nargs="*",
)
parser.add_argument(
    "--output",
    type=Path,
    required=True
)
args = parser.parse_args()

pathlist_chi2est_data:list[Path] = args.data
outpath:Path = args.output
outpath.parent.mkdir(parents=True,exist_ok=True)

dfs_output:list[pd.DataFrame] = [
    fr.read_csv(path)
    for path in pathlist_chi2est_data
]

df_output = pd.concat(dfs_output)
fw.write_csv(
    df_output,outpath
)
