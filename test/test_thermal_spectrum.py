# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false

import astropy.units as u
from module.models import SynchrotronSpectrum
import pandas as pd
import numpy as np
from module.utilities import filewriters as fw
from pathlib import Path

def compute(input_params:dict):
    metadata = {}

    ss = SynchrotronSpectrum(**input_params)
    df = pd.DataFrame({
        "nu":ss.nu_array,
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
    "a_wind": 1.0e+07*u.g/u.cm,
    "t": 1.0*u.min,
    "nu_array": u.Quantity(np.logspace(3,15,512),u.Hz)
}

compute(params)
