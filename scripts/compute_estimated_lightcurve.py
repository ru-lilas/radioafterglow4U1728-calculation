import argparse
from typing import Any
from module import dataframe_utils, fetch_numerical_table, input_reader
from pathlib import Path
import numpy as np
from module.utilities import filereaders as fr
from module.utilities import filewriters as fw
from module.utilities import build_nparray
import pandas as pd
from module.utilities import quantity_data
from module import dataframe_processors as dfp
from module import compute_lightcurve
from module.strenums import KeyNames
import astropy.units as u

parser = argparse.ArgumentParser()

parser.add_argument(
    "--output",
    type=Path,
    required=True
)
parser.add_argument(
    "--estimated_parameters",
    type=Path,
)
parser.add_argument(
    "--physical_parameters",
    type=Path,
    required=True
)
parser.add_argument(
    "--lightcurve_config",
    type=Path,
    required=True
)
parser.add_argument(
    "--table_integral",
    type=Path,
    required=True
)
args = parser.parse_args()

outpath:Path = args.output
outpath.parent.mkdir(parents=True,exist_ok=True)

path_phys_params:Path = args.physical_parameters
data_phys_params = input_reader.read_physical_parameters(path_phys_params)

table_integral = fetch_numerical_table.fetch_numerical_table_path(args.table_integral)

lcconfpath:Path = args.lightcurve_config
conf_lc = input_reader.InputReader.read(lcconfpath,input_reader.LightcurveConfigure)
lcconf = fr.read_yaml_pyyaml(lcconfpath)

inpath:Path = args.estimated_parameters
df_params_estimated = fr.read_csv(inpath)
print(df_params_estimated.head())
metadata_params_estimated = fr.read_keyvalue(inpath)

t = quantity_data.QuantityData(
    conf_lc.t_value_arr.nparr,
    conf_lc.t_unit
)

d_value = np.asarray(data_phys_params.distance.value,dtype=np.float64)
d_unit = data_phys_params.distance.unit
d_src = quantity_data.QuantityData(
    value = d_value,
    unit = d_unit
)
nu_values = np.asarray(df_params_estimated[KeyNames.NU],dtype=np.float64)
# nu_unit = metadata_params_estimated[KeyNames.NU_UNIT]
nu_unit = "GHz"

lnu_theta_values = dfp.convert_ndarray(df_params_estimated,KeyNames.LNU_THETA)
# lnu_unit = metadata_params_estimated[KeyNames.LNU_UNIT]
lnu_unit = "erg Hz-1 s-1"

phi_theta_values = np.asarray(df_params_estimated[KeyNames.PHI_THETA],dtype=np.float64)
# phi_unit = metadata_params_estimated[KeyNames.PHI_UNIT]
phi_unit = "GHz s"

tau_theta = np.asarray(df_params_estimated[KeyNames.TAU_THETA],dtype=np.float64)
doppler_delta = np.asarray(df_params_estimated[KeyNames.DOPPLER_DELTA],dtype=np.float64)

fnu_unit = lcconf[KeyNames.FNU_UNIT]

a_wind_arr = dataframe_utils.extract_column_as_ndarray(df_params_estimated,KeyNames.A_WIND)
beta_sh_arr = dataframe_utils.extract_column_as_ndarray(df_params_estimated,KeyNames.BETA_SH)
chi2_arr = dataframe_utils.extract_column_as_ndarray(df_params_estimated,KeyNames.CHI2)

dfs: list[pd.DataFrame] = []
for i,nu in enumerate(nu_values):
    nu_q = quantity_data.QuantityData(
        value = nu,
        unit = nu_unit
    )
    lnu_theta_q = quantity_data.QuantityData(
        value = lnu_theta_values[i],
        unit = lnu_unit
    )
    phi_theta_q = quantity_data.QuantityData(
        value = phi_theta_values[i],
        unit = phi_unit
    )


    lc = compute_lightcurve.LightcurveCalculation(
        t=t,
        nu = nu_q,
        table_integral=table_integral
    )
    tau_theta_q = tau_theta[i]
    doppler_delta_q = doppler_delta[i]

    fnu_quantity = lc.fnu_with_doppler(
        phi_theta=phi_theta_q,
        lnu_theta=lnu_theta_q,
        tau_theta=tau_theta_q,
        d_src=d_src,
        doppler_delta=doppler_delta_q
    )

    phi = u.Quantity(nu_q.quantity*t.quantity)
    phi_value = phi.to_value(phi_unit)

    dfs.append(pd.DataFrame({
        KeyNames.NU: nu,
        KeyNames.T: t.value,
        KeyNames.PHI: phi_value,
        KeyNames.FNU_NET: fnu_quantity.unit_convert(fnu_unit),
        KeyNames.A_WIND: a_wind_arr[i],
        KeyNames.BETA_SH: beta_sh_arr[i],
        "chi2": chi2_arr[i],
        KeyNames.REDUCED_CHI2: dataframe_utils.extract_column_as_ndarray(df_params_estimated,KeyNames.REDUCED_CHI2)[i]
    }))
metadata:dict[str,Any] = {
    KeyNames.T_UNIT: t.unit,
    KeyNames.LNU_UNIT: lnu_unit,
    KeyNames.NU_UNIT: nu_unit,
    KeyNames.FNU_UNIT: fnu_unit,
    KeyNames.D_VALUE: d_value,
    KeyNames.D_UNIT: d_unit,
    KeyNames.EPS_B: metadata_params_estimated[KeyNames.EPS_B],
    KeyNames.EPS_TH: metadata_params_estimated[KeyNames.EPS_TH],
    "mu": metadata_params_estimated["mu"],
    "mu_e": metadata_params_estimated["mu_e"],
    "a_wind_unit": metadata_params_estimated["a_wind_unit"]
}
df = pd.concat(dfs)
fw.write_csv_with_params(df,metadata,outpath)
