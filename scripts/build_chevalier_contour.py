import argparse
from pathlib import Path
from module.fetch_numerical_table import fetch_peak_table_path
from module.utilities import filereaders as fr
from module.utilities import filewriters as fw
import numpy as np
from module.plot import processor
from module.strenums import KeyNames
from module import dataframe_utils as dfutils, quantity_converter
import astropy.units as u
from typing import cast

parser = argparse.ArgumentParser()
parser.add_argument(
    "input",
    type=Path,
)
parser.add_argument(
    "--peak_table",
    type=Path,
    required=True
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
tablepath:Path = args.peak_table

df_input = fr.read_csv(inpath)
metadata = fr.read_keyvalue(inpath)

table = fetch_peak_table_path(tablepath)
tau_theta_arr = dfutils.extract_column_as_ndarray(df_input,KeyNames.TAU_THETA)

# df_output = df_input.assign(
#     xm_peak=table.calculate_xm_peak(tau_theta_arr),
#     lambda_peak=table.calculate_lambda_peak(tau_theta_arr),
# )

d_src_value = metadata[KeyNames.D_VALUE]
d_src_unit = metadata[KeyNames.D_UNIT]
d_src = u.Quantity(d_src_value,d_src_unit)

phi_theta_value = dfutils.extract_column_as_ndarray(df_input,KeyNames.PHI_THETA)
phi_unit = metadata[KeyNames.PHI_UNIT]
phi_theta = u.Quantity(phi_theta_value,phi_unit)

lnu_theta_value = dfutils.extract_column_as_ndarray(df_input,KeyNames.LNU_THETA)
lnu_unit = metadata[KeyNames.LNU_UNIT]
lnu_theta = u.Quantity(lnu_theta_value,lnu_unit)

doppler_delta = dfutils.extract_column_as_ndarray(df_input,KeyNames.DOPPLER_DELTA)

xm_peak = table.calculate_xm_peak(tau_theta_arr)
lambda_peak = table.calculate_lambda_peak(tau_theta_arr)

phi_peak = cast(u.Quantity,phi_theta*xm_peak)
lnu_peak = cast(u.Quantity,lnu_theta*lambda_peak)

fnu_unit = metadata[KeyNames.FNU_UNIT]
fnu_peak_no_doppler = quantity_converter.lnu_into_fnu(
    lnu = lnu_peak,
    distance = d_src
)

fnu_peak_with_doppler = cast(u.Quantity,doppler_delta**3*fnu_peak_no_doppler)
# phi_peak_arr = processor.calcualte_product_two_columns(
#     df=df_output,
#     columns=("phi_theta","xm_peak")
# )
# l_peak_arr = processor.calcualte_product_two_columns(
#     df=df_output,
#     columns=("lnu_theta","lambda_peak")
# )
df_output = df_input.assign(
    phi_peak=phi_peak.to_value(phi_unit),
    fnu_peak=fnu_peak_with_doppler.to_value(fnu_unit)
)
fw.write_csv_with_params(df_output,metadata,outpath)

