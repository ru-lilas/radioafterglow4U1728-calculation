# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false

import astropy.units as u
from module.models import SynchrotronSpectrum
import pandas as pd
import numpy as np
from module.utilities import filewriters as fw
from pathlib import Path

def compute(input_params:dict):
    metadata = {
        key: value
        for key, value in input_params.items()
        if key != "nu_array_value"
    }

    ss = SynchrotronSpectrum(**input_params)
    metadata["r_value"] = ss.r.value
    metadata["r_unit"] = ss.r.unit
    metadata["n_wind_value"] = ss.n_wind.value
    metadata["n_wind_unit"] = str(ss.n_wind.unit)
    metadata["b_mag_value"] = ss.b_mag.value
    metadata["b_mag_unit"] = str(ss.b_mag.unit)
    metadata["theta_e"] = ss.theta_e

    df = pd.DataFrame({
        "nu":ss.nu_array_quantity,
        "chi":ss.chi,
        "x_m":ss.xm,
        "jnu_dimless":ss.j_nu_th_dimless,
        "anu_dimless":ss.a_nu_th_dimless,
        "snu_dimless":ss.S_nu_th_dimless,
        "lnu":ss.lnu_th
    })
    # theta_fmt = int(ss.theta*10)
    filepath = Path(f"data/test/thermal_spectrum_000.csv")
    fw.write_csv_with_params(df,metadata,filepath)
    print(f"output {filepath}")

params = {
    "eps_th": 1.0,
    "eps_B": 1.0e-1,
    "mu": 0.62,
    "mu_e": 1.18,
    "beta_sh": 0.1,
    "a_wind_value": 1.0e+07,
    "a_wind_unit": "g/cm",
    "t_value": 1.0,
    "t_unit": "min", 
    "nu_array_value": np.logspace(3,15,512),
    "nu_array_unit": "Hz"
}

compute(params)
