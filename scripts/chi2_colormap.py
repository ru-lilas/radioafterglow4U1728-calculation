from module import dataframe_utils
from module.strenums import KeyNames
from module.utilities import filereaders as fr
from module.utilities import filewriters as fw
from pathlib import Path
import argparse
import pandas as pd
from module.strenums import KeyNames

parser = argparse.ArgumentParser()

parser.add_argument(
    "parameter_table",
    type=Path,
)
parser.add_argument(
    "--chi2_table",
    type=Path,
)
parser.add_argument(
    "-o",
    "--output",
    type=Path,
    required=True
)
args = parser.parse_args()

parameter_path:Path = args.parameter_table
chi2_path:Path = args.chi2_table

outpath:Path = args.output
outpath.parent.mkdir(parents=True,exist_ok=True)

df_param = fr.read_csv_within_idx(parameter_path)
metadata_param = fr.read_keyvalue(parameter_path)
df_chi2 = fr.read_csv(chi2_path)
metadata_chi2 = fr.read_keyvalue(chi2_path)

chi2_nu_data = dataframe_utils.build_dfs_grouped(
    df_chi2,group_by=KeyNames.NU
)
metadata_output = {
    **metadata_param,
    **metadata_chi2
}
dfs_output = []
for nu,df_chi2_nu_raw in chi2_nu_data:
    df_chi2_nu = df_chi2_nu_raw.set_index("idx")
    dfs_output.append(df_param.join(df_chi2_nu))

df_output = pd.concat(dfs_output)
fw.write_csv_with_params(df_output,metadata_output,outpath)
