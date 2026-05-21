from module.models import SynchrotronSpectrum
import pandas as pd
from module.utilities import filewriters as fw
from pathlib import Path

def compute(input_params:dict):

    print("calculating spectrum...")
    ss = SynchrotronSpectrum(**input_params)

    metadata = build_metadata(input_params,ss)
    df = build_tabledata(ss)

    return metadata,df

def build_metadata(input_params:dict,ss:SynchrotronSpectrum):
    print("building metadata...")
    metadata = {
        key: value
        for key, value in input_params.items()
        if key != "nu_array_value"
    }

    metadata["r_value"] = ss.r.value
    metadata["r_unit"] = ss.r.unit
    metadata["n_wind_value"] = ss.n_wind.value
    metadata["n_wind_unit"] = ss.n_wind.unit
    metadata["b_mag_value"] = ss.b_mag.value
    metadata["b_mag_unit"] = ss.b_mag.unit
    metadata["theta_e"] = ss.theta_e
    metadata["nu_B_value"] = ss.nu_B.value
    metadata["nu_crit_value"] = ss.nu_crit.value
    metadata["j0_value"] = ss.j0.value
    metadata["j0_unit"] = ss.j0.unit
    metadata["a0_value"] = ss.a0.value
    metadata["a0_unit"] = ss.a0.unit

    return metadata

def build_tabledata(ss:SynchrotronSpectrum)->pd.DataFrame:
    print("building table-format data...")
    df = pd.DataFrame({
        "nu":ss.nu_array_quantity,
        "chi":ss.chi,
        "x_m":ss.xm,
        "jnu_dimless":ss.j_nu_th_dimless,
        "jnu_th":ss.j_nu_th.value,
        "anu_dimless":ss.a_nu_th_dimless,
        "anu_th":ss.a_nu_th.value,
        "snu_dimless":ss.S_nu_th_dimless,
        "lnu":ss.lnu_th,
        "tau_nu":ss.tau_nu
    })
    return df
