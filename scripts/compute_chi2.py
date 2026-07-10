from typing import Any, cast
import argparse
from pathlib import Path

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

def filter_df_obs_time_window(df_obs:pd.DataFrame,conf_filter:dict[str,Any]):
    return filter_df_value_window(
        df_obs,
        column_name = KeyNames.T,
        min = conf_filter[EstimationConfigNames.MIN],
        max = conf_filter[EstimationConfigNames.MAX]
    )

def calculate_chi2(
    y_model:NDArray[np.float64],
    y_obs:NDArray[np.float64],
    y_err:NDArray[np.float64],
):
    chi_arr = (y_model - y_obs)/y_err
    return np.sum(chi_arr**2).item()


parser = argparse.ArgumentParser()

parser.add_argument(
    "parameter_table",
    type=Path,
)
parser.add_argument(
    "--integral_table",
    type=Path,
    required=True
)
parser.add_argument(
    "--observation_lc",
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

param_table_path:Path = args.parameter_table
metadata_param_table = fr.read_keyvalue(param_table_path)
df_param_table = fr.read_csv_within_idx(param_table_path)

integral_table_path:Path = args.integral_table
data_integral = fetch_numerical_table_path(integral_table_path)

obspath:Path = args.observation_lc
df_obs_raw = fr.read_csv(obspath)
metadata_obs = fr.read_keyvalue(obspath)

confpath:Path = args.config
conf_sampling = fr.read_yaml(confpath)

outpath:Path = args.output
outpath.parent.mkdir(parents=True,exist_ok=True)

d_src = QuantityData(
    value = np.asarray(metadata_param_table[KeyNames.D_VALUE],dtype=np.float64),
    unit = metadata_param_table[KeyNames.D_UNIT]
)

df_obs = filter_df_obs_time_window(
    df_obs_raw,conf_sampling
)

phi_unit = metadata_param_table[KeyNames.PHI_UNIT]
lnu_unit = metadata_param_table[KeyNames.LNU_UNIT]
fnu_unit = metadata_obs[KeyNames.FNU_UNIT]

nu_unit = metadata_obs[KeyNames.NU_UNIT]
bestfit_rows:list[pd.Series] = []
df_list = []

m_parameter = 2
for nu_obs, df_obs_nu in df_obs.groupby(KeyNames.NU,sort=False):
    df_obs_nu = df_obs_nu.reset_index(drop=True)
    n_sample = len(df_obs_nu)
    num_freedom = n_sample - m_parameter
    nu_obs = np.asarray(cast(float,nu_obs),dtype=np.float64)
    nu = QuantityData(nu_obs,nu_unit)
    print(f"nu = {nu_obs:.2e} {nu_unit}")
    t_value = np.asarray(df_obs_nu[KeyNames.T],dtype=np.float64)
    t_unit = metadata_obs[KeyNames.T_UNIT]
    fnu_net_obs = QuantityData(
        np.asarray(df_obs_nu[KeyNames.FNU_NET],dtype=np.float64),
        fnu_unit
    )
    fnu_err = QuantityData(
        np.asarray(df_obs_nu[KeyNames.FNU_ERR],dtype=np.float64),
        fnu_unit
    )

    lc = compute_lightcurve.Lightcurve(
        t = QuantityData(t_value,t_unit),
        nu = nu,
        table_integral = data_integral
    )

    # create array of chi**2 in advance
    chi2_arr_with_doppler = np.empty(len(df_param_table),dtype=np.float64)

    # observation
    y_sample=fnu_net_obs.to_ndarray(fnu_unit)
    y_sample_err=fnu_err.to_ndarray(fnu_unit)

    phi_theta_arr = dataframe_utils.extract_column_as_ndarray(df_param_table,KeyNames.PHI_THETA)
    lnu_theta_arr = dataframe_utils.extract_column_as_ndarray(df_param_table,KeyNames.LNU_THETA)
    tau_theta_arr = dataframe_utils.extract_column_as_ndarray(df_param_table,KeyNames.TAU_THETA)
    doppler_delta_arr = dataframe_utils.extract_column_as_ndarray(df_param_table,KeyNames.DOPPLER_DELTA)
    idx_arr = df_param_table.index.to_numpy()

    rows = []
    for i,idx in enumerate(tqdm(idx_arr,desc="Calculating chi2")):
        phi_theta_value = phi_theta_arr[i]
        lnu_theta_value = lnu_theta_arr[i]
        tau_theta_value = tau_theta_arr[i]
        doppler_delta = doppler_delta_arr[i]

        phi_theta = QuantityData(
            value = phi_theta_value,
            unit = phi_unit
        )
        lnu_theta = QuantityData(
            value = lnu_theta_value,
            unit = lnu_unit
        )

        fnu_model_with_doppler = lc.fnu_with_doppler(
            phi_theta=phi_theta,
            lnu_theta=lnu_theta,
            tau_theta=tau_theta_value,
            d_src=d_src,
            doppler_delta=doppler_delta
        )

        chi2_arr_with_doppler[i] = (calculate_chi2(
            y_model=fnu_model_with_doppler.to_ndarray(fnu_unit),
            y_obs=y_sample,
            y_err=y_sample_err
        ))
        reduced_chi2 = chi2_arr_with_doppler[i]/float(num_freedom)

        rows.append({
            "idx": idx,
            KeyNames.NU: nu.value,
            KeyNames.CHI2: chi2_arr_with_doppler[i],
            KeyNames.REDUCED_CHI2: reduced_chi2
        })

    df_list.append(
        pd.DataFrame(rows)
    )

df_output = pd.concat(df_list,ignore_index=True)
metadata_output = {
    KeyNames.NU_UNIT: nu_unit,
}

fw.write_csv_with_params(df_output,metadata_output,outpath)
