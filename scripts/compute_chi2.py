from functools import cached_property
from typing import Any, cast
import argparse
from pathlib import Path
from math import isclose
from dataclasses import dataclass

from module.utilities import filereaders as fr
from dacite import from_dict
from module import compute_lightcurve, input_dataclasses, mydataclasses
import numpy as np
import pandas as pd

from tqdm import tqdm
from module.dataframe_processors import filter_df_value_window
from module.utilities import filewriters as fw
from module.strenums import KeyNames, EstimationConfigNames, LightcurveColumns
from module.strenums import Chi2Columns
from module.fetch_numerical_table import fetch_numerical_table_path
from module.utilities.quantity_data import QuantityData
from module import compute_lightcurve, dataframe_utils, input_reader
from module import mystatistics
from module import observation
from functools import cached_property
import astropy.units as u
from module.parameter_table import GeneralInputs

def filter_df_obs_time_window(df_obs:pd.DataFrame,conf_filter:dict[str,Any]):
    return filter_df_value_window(
        df_obs,
        column_name = KeyNames.T,
        min = conf_filter[EstimationConfigNames.MIN],
        max = conf_filter[EstimationConfigNames.MAX]
    )

parser = argparse.ArgumentParser()

parser.add_argument(
    "--parameter_table",
    type=Path,
    required=True
)
parser.add_argument(
    "--integral_table",
    type=Path,
    required=True
)
parser.add_argument(
    "--obs_lc",
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

path_conf:Path = args.config
path_param_table:Path = args.parameter_table
path_integral_table:Path = args.integral_table
path_obslc:Path = args.obs_lc
outpath:Path = args.output
outpath.parent.mkdir(parents=True,exist_ok=True)

conf = GeneralInputs.from_yaml(path_conf)
conf_fitting = conf.chi2fitting
conf_sampling = conf_fitting.sampling

obslc_general = observation.LongformatLightcurve.from_csv(path_obslc)
obslc = obslc_general.extract_lightcurve(conf_sampling.nu.value)

# calculate model lightcurves
lc_conf = conf_fitting.model
table_integral = compute_lightcurve.ThermalSynchrotronTable.from_csv(path_integral_table)
lc_model = compute_lightcurve.ThermalSynchrotron(table_integral)
lc_inputs = compute_lightcurve.build_inputs(path_param_table)

for lc_input in lc_inputs:
    lc = compute_lightcurve.compute(
        config=lc_conf,
        model=lc_model,
        input=lc_input
    )
    # lc_binned = lc.bin_average(
    #     bin_width=metadata_obs.bin_width,
    # )

# for nu_obs, df_obs_nu in dfset_obs.items():
#     df_obs_nu = df_obs_nu.reset_index(drop=True)
#     n_sample = len(df_obs_nu)
#     ndof = n_sample - n_param
#     if ndof <= 0:
#         raise ValueError(
#             f"Degrees of freedom must be positive (ndof={ndof}, "
#             f"n_data={n_sample}, n_param={n_param})."
#         )
#     nu_obs = np.asarray(nu_obs,dtype=np.float64)
#     nu = QuantityData(nu_obs,nu_unit)
#     print(f"nu = {nu_obs:.2e} {nu_unit}")
#
#     t_value = dataframe_utils.extract_column_as_ndarray(df_obs_nu,LightcurveColumns.T)
#     t_unit = metadata_obs[KeyNames.T_UNIT]
#     fnu_net_obs = QuantityData(
#         dataframe_utils.extract_column_as_ndarray(df_obs_nu,LightcurveColumns.FNU_NET),
#         fnu_unit
#     )
#     fnu_err = QuantityData(
#         dataframe_utils.extract_column_as_ndarray(df_obs_nu,LightcurveColumns.FNU_NET_ERR),
#         fnu_unit
#     )
#
#     lc = compute_lightcurve.LightcurveCalculation(
#         t = QuantityData(t_value,t_unit),
#         nu = nu,
#         table_integral = data_integral
#     )
#
#     # create array of chi**2 in advance
#     chi2_arr_with_doppler = np.empty(len(df_param_table),dtype=np.float64)
#
#     # observation
#     y_sample=fnu_net_obs.to_ndarray(fnu_unit)
#     y_sample_err=fnu_err.to_ndarray(fnu_unit)
#
#     phi_theta_arr = dataframe_utils.extract_column_as_ndarray(df_param_table,KeyNames.PHI_THETA)
#     lnu_theta_arr = dataframe_utils.extract_column_as_ndarray(df_param_table,KeyNames.LNU_THETA)
#     tau_theta_arr = dataframe_utils.extract_column_as_ndarray(df_param_table,KeyNames.TAU_THETA)
#     doppler_delta_arr = dataframe_utils.extract_column_as_ndarray(df_param_table,KeyNames.DOPPLER_DELTA)
#     idx_arr = df_param_table.index.to_numpy()
#
#     rows = []
#     for i,idx in enumerate(tqdm(idx_arr,desc="Calculating chi2")):
#         phi_theta_value = phi_theta_arr[i]
#         lnu_theta_value = lnu_theta_arr[i]
#         tau_theta_value = tau_theta_arr[i]
#         doppler_delta = doppler_delta_arr[i]
#
#         phi_theta = QuantityData(
#             value = phi_theta_value,
#             unit = phi_unit
#         )
#         lnu_theta = QuantityData(
#             value = lnu_theta_value,
#             unit = lnu_unit
#         )
#
#         fnu_model_with_doppler = lc.fnu_with_doppler(
#             phi_theta=phi_theta,
#             lnu_theta=lnu_theta,
#             tau_theta=tau_theta_value,
#             d_src=d_src,
#             doppler_delta=doppler_delta
#         )
#
#         chi2_arr_with_doppler[i] = mystatistics.calculate_chi2(
#             x_model=fnu_model_with_doppler.to_ndarray(fnu_unit),
#             x_data=y_sample,
#             sigma=y_sample_err
#         )
#         reduced_chi2 = mystatistics.calculate_reduced_chi2(
#             chi2 = chi2_arr_with_doppler[i],
#             ndof = ndof
#         )
#
#         rows.append({
#             "idx": idx,
#             KeyNames.NU: nu.value,
#             KeyNames.CHI2: chi2_arr_with_doppler[i],
#             KeyNames.REDUCED_CHI2: reduced_chi2,
#             Chi2Columns.NPARAM: n_param,
#             Chi2Columns.NSAMPLE: n_sample,
#             Chi2Columns.NDOF: ndof
#         })
#     df = pd.DataFrame(rows).astype({
#         Chi2Columns.NPARAM: "int64",
#         Chi2Columns.NSAMPLE: "int64",
#         Chi2Columns.NDOF: "int64"
#     })
#     df_list.append(
#         pd.DataFrame(rows)
#     )
#
# df_output = pd.concat(df_list,ignore_index=True)
# metadata_output = {
#     KeyNames.NU_UNIT: nu_unit,
# }
#
# fw.write_csv_with_params(df_output,metadata_output,outpath)
