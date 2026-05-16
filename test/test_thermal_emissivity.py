from module import emissivity
from numpy import logspace
import pandas as pd
from module.utilities import filewriters as fw
from pathlib import Path

def compute_jnu(theta:float):
    chi = logspace(-6.0,7.0,256)

    jnu = emissivity.j_th(theta,chi)
    x_m = emissivity.convert_xm(theta,chi)

    df = pd.DataFrame({
        "chi":chi,
        "x_m":x_m,
        "jnu":jnu,
    })
    theta_fmt = int(theta*10)
    fw.write_csv_with_params(df,{"theta":theta},Path(f"data/test/thermal_emissivity_theta{theta_fmt:03d}.csv"))

compute_jnu(theta=5.0)
compute_jnu(theta=1.0)
compute_jnu(theta=0.5)
