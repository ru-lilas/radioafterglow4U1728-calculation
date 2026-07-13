import pandas as pd
from typing import Any
from module import quantity_converter
from module.models import InputParameters
import astropy.units as u
from itertools import product
from tqdm import tqdm
from module.strenums import KeyNames
from module.utilities import build_nparray

def _table_row(
    beta_sh: float,
    a_wind_value: float,
    fixed: dict[str, Any],
    units_designated: dict[str,str]
) -> dict[str,Any]:

    params = InputParameters(
        **fixed,
        beta_sh=beta_sh,
        a_wind_value=a_wind_value,
        a_wind_unit=units_designated["a_wind_unit"]
    )
    theta_e = params.theta
    phi_unit = u.Unit(units_designated["phi_unit"])
    lnu_unit = u.Unit(units_designated["lnu_unit"])
    phi_theta_value = params.phi_theta.to_value(phi_unit)
    lnu_theta_value = params.l_theta.to_value(lnu_unit)
    doppler_delta = quantity_converter.beta_into_doppler_delta(beta_sh)

    return {
        KeyNames.BETA_SH: beta_sh,
        KeyNames.A_WIND: a_wind_value,
        "theta": theta_e,
        KeyNames.PHI_THETA: phi_theta_value,
        KeyNames.LNU_THETA: lnu_theta_value,
        KeyNames.TAU_THETA: params.tau_theta,
        KeyNames.DOPPLER_DELTA: doppler_delta,
    }

def table(
    config_data:dict[str,Any]
)->pd.DataFrame:
    a_wind_arr = build_nparray.log(config_data["a_wind_arr"])
    beta_sh_arr = build_nparray.log(config_data["beta_sh_arr"])

    table_list:list[dict] = []
    for beta,a_wind in tqdm(product(beta_sh_arr,a_wind_arr),total=beta_sh_arr.size*a_wind_arr.size,desc="table"):
        table_row = _table_row(
            beta,
            a_wind,
            config_data["microphysics"],
            config_data["units"]
        )
        table_list.append(table_row)

    return pd.DataFrame(table_list)
