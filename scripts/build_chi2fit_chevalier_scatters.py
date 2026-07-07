import argparse
from pathlib import Path
from module.utilities import filereaders as fr
from module.utilities import filewriters as fw
from module.strenums import KeyNames, ChevalierContourNames
from module import dataframe_processors as dfp

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

inpath:Path = args.input
outpath:Path = args.output
outpath.parent.mkdir(parents=True,exist_ok=True)

metadata_input = fr.read_keyvalue(inpath)
df_input = fr.read_csv(inpath)

phi_peaks, fnu_net_peaks = dfp.extract_maximum(
    df_input,KeyNames.PHI,KeyNames.FNU_NET
)

df_output = df_input.copy()
df_output[ChevalierContourNames.PHI_PEAK] = phi_peaks
df_output[ChevalierContourNames.FNU_NET_PEAK] = fnu_net_peaks

fw.write_csv_with_params(df_output,metadata_input,outpath)
