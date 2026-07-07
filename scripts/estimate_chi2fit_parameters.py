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
from module import compute_lightcurve

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
df_param_table = fr.read_csv(param_table_path)

integral_table_path:Path = args.integral_table
data_integral = fetch_numerical_table_path(integral_table_path)

obspath:Path = args.observation_lc
df_obs_raw = fr.read_csv(obspath)
metadata_obs = fr.read_keyvalue(obspath)

confpath:Path = args.config
conf = fr.read_yaml(confpath)

outpath:Path = args.output
outpath.parent.mkdir(parents=True,exist_ok=True)

d_src = QuantityData(
    value = np.asarray(metadata_param_table[KeyNames.D_VALUE],dtype=np.float64),
    unit = metadata_param_table[KeyNames.D_UNIT]
)

df_obs = filter_df_obs_time_window(
    df_obs_raw,conf[EstimationConfigNames.OBS_T_WINDOW]
)

phi_unit = metadata_param_table[KeyNames.PHI_UNIT]
lnu_unit = metadata_param_table[KeyNames.LNU_UNIT]
fnu_unit = metadata_obs[KeyNames.FNU_UNIT]

nu_unit = metadata_obs[KeyNames.NU_UNIT]
bestfit_rows:list[pd.Series] = []
for nu_obs, df_obs_nu in df_obs.groupby(KeyNames.NU,sort=False):
    df_obs_nu = df_obs_nu.reset_index(drop=True)
    nu_obs = np.asarray(cast(float,nu_obs),dtype=np.float64)
    nu = QuantityData(nu_obs,nu_unit)
    print(f"nu = {nu_obs:.2e} {nu_unit}")
    print(df_obs_nu)
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
    chi2_arr_no_doppler = np.empty(len(df_param_table),dtype=np.float64)
    chi2_arr_with_doppler = np.empty(len(df_param_table),dtype=np.float64)

    # observation
    y_obs=fnu_net_obs.to_ndarray(fnu_unit)
    y_err=fnu_err.to_ndarray(fnu_unit)

    for idx,(phi_theta_value,lnu_theta_value,tau_theta,doppler_delta) \
        in enumerate(tqdm(
            df_param_table[[KeyNames.PHI_THETA,KeyNames.LNU_THETA,KeyNames.TAU_THETA,KeyNames.DOPPLER_DELTA]].itertuples(index=False,name=None),
            total=len(df_param_table),
            desc="Calculating chi2",
        )):
        phi_theta = QuantityData(
            value = phi_theta_value,
            unit = phi_unit
        )
        lnu_theta = QuantityData(
            value = lnu_theta_value,
            unit = lnu_unit
        )

        # fnu_model_no_doppler = lc.fnu(
        #     phi_theta=phi_theta,
        #     lnu_theta=lnu_theta,
        #     tau_theta=tau_theta,
        #     d_src=d_src
        # )
        fnu_model_with_doppler = lc.fnu_with_doppler(
            phi_theta=phi_theta,
            lnu_theta=lnu_theta,
            tau_theta=tau_theta,
            d_src=d_src,
            doppler_delta=doppler_delta
        )

        # chi2_arr_no_doppler[idx] = (calculate_chi2(
        #     y_model=fnu_model_no_doppler.to_ndarray(fnu_unit),
        #     y_obs=y_obs,
        #     y_err=y_err
        # ))
        chi2_arr_with_doppler[idx] = (calculate_chi2(
            y_model=fnu_model_with_doppler.to_ndarray(fnu_unit),
            y_obs=y_obs,
            y_err=y_err
        ))

    # idx_best_no_doppler = np.argmin(chi2_arr_no_doppler)
    idx_best_with_doppler = np.argmin(chi2_arr_with_doppler)
    # chi2_min_no_doppler = chi2_arr_no_doppler[idx_best_no_doppler]
    chi2_min_with_doppler = chi2_arr_with_doppler[idx_best_with_doppler]

    # param_best = pd.Series(df_param_table.iloc[idx_best_no_doppler])
    param_best = pd.Series(df_param_table.iloc[idx_best_with_doppler])
    param_best[KeyNames.NU] = nu.value
    # param_best[KeyNames.CHI2] = chi2_min_no_doppler
    param_best[KeyNames.CHI2] = chi2_min_with_doppler

    bestfit_rows.append(param_best)

df_output = pd.DataFrame(bestfit_rows)
metadata_output = {
    **metadata_param_table,
    KeyNames.NU_UNIT: nu_unit
}

fw.write_csv_with_params(df_output,metadata_output,outpath)
