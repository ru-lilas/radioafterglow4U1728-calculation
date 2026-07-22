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
    "--chi2_estimated",
    type=Path,
    required=True
)
parser.add_argument(
    "--sampling_config",
    type=Path,
    required=True
)
parser.add_argument(
    "--output",
    type=Path,
    required=True
)
args = parser.parse_args()

path_chi2_est:Path = args.chi2_estimated
path_sampling:Path = args.sampling_config
outpath:Path = args.output
outpath.parent.mkdir(parents=True,exist_ok=True)

tw = InputReader.read(path_sampling,SAMPLING)
df = fr.read_csv(path_chi2_est)
metadata = fr.read_keyvalue(path_chi2_est)

df[SamplingConfigNames.T_MIN] = tw.min
df[SamplingConfigNames.T_MAX] = tw.max
df[SamplingConfigNames.T_UNIT] = tw.unit

fw.write_csv_with_params(df,metadata,outpath)
