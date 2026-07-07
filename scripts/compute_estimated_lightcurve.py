import argparse
from typing import Any
from module import fetch_numerical_table
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
    "input",
    type=Path,
)
parser.add_argument(
    "--config",
    type=Path,
    required=True
)
parser.add_argument(
    "--table_integral",
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

table_integral = fetch_numerical_table.fetch_numerical_table_path(args.table_integral)

confpath:Path = args.config
config_calc = fr.read_yaml(confpath)

inpath:Path = args.input
df_params_estimated = fr.read_csv(inpath)
metadata_params_estimated = fr.read_keyvalue(inpath)

t = quantity_data.QuantityData(
    build_nparray.log(config_calc["t_value_arr"]),
    config_calc["t_unit"]
)

d_value = np.asarray(metadata_params_estimated[KeyNames.D_VALUE],dtype=np.float64)
d_unit = metadata_params_estimated[KeyNames.D_UNIT]
d_src = quantity_data.QuantityData(
    value = d_value,
    unit = d_unit
)
nu_values = np.asarray(df_params_estimated[KeyNames.NU],dtype=np.float64)
nu_unit = metadata_params_estimated[KeyNames.NU_UNIT]

lnu_theta_values = dfp.convert_ndarray(df_params_estimated,KeyNames.LNU_THETA)
lnu_unit = metadata_params_estimated[KeyNames.LNU_UNIT]

phi_theta_values = np.asarray(df_params_estimated[KeyNames.PHI_THETA],dtype=np.float64)
phi_unit = metadata_params_estimated[KeyNames.PHI_UNIT]

tau_theta = np.asarray(df_params_estimated[KeyNames.TAU_THETA],dtype=np.float64)

fnu_unit = config_calc[KeyNames.FNU_UNIT]

a_wind_arr = np.asarray(df_params_estimated[KeyNames.A_WIND],dtype=np.float64)
beta_sh_arr = np.asarray(df_params_estimated[KeyNames.BETA_SH],dtype=np.float64)

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


    lc = compute_lightcurve.Lightcurve(
        t=t,
        nu = nu_q,
        table_integral=table_integral
    )
    tau_theta_q = tau_theta[i]
    fnu_quantity = lc.fnu(
        phi_theta=phi_theta_q,
        lnu_theta=lnu_theta_q,
        tau_theta=tau_theta_q,
        d_src=d_src
    )

    phi = u.Quantity(nu_q.quantity*t.quantity)
    phi_value = phi.to_value(phi_unit)

    dfs.append(pd.DataFrame({
        KeyNames.NU: nu,
        KeyNames.T: t.value,
        KeyNames.PHI: phi_value,
        KeyNames.FNU_NET: fnu_quantity.unit_convert(fnu_unit),
        KeyNames.A_WIND: a_wind_arr[i],
        KeyNames.BETA_SH: beta_sh_arr[i]
    }))
metadata:dict[str,Any] = {
    KeyNames.T_UNIT: t.unit,
    KeyNames.LNU_UNIT: lnu_unit,
    KeyNames.NU_UNIT: nu_unit,
    KeyNames.FNU_UNIT: fnu_unit,
    KeyNames.D_VALUE: d_value,
    KeyNames.D_UNIT: d_unit,
    "eps_B": metadata_params_estimated["eps_B"],
    "eps_th": metadata_params_estimated["eps_th"],
    "mu": metadata_params_estimated["mu"],
    "mu_e": metadata_params_estimated["mu_e"],
    "a_wind_unit": metadata_params_estimated["a_wind_unit"]
}
df = pd.concat(dfs)
fw.write_csv_with_params(df,metadata,outpath)
