from module.models import SynchrotronSpectrum
import pandas as pd
import numpy as np
from module.utilities import filewriters as fw
from pathlib import Path

def compute(theta:float,chi:np.ndarray):

    ss = SynchrotronSpectrum(theta,chi)

    df = pd.DataFrame({
        "chi":ss.chi,
        "x_m":ss.xm,
        "jnu":ss.jnu_th,
        "anu":ss.anu_th,
    })
    theta_fmt = int(theta*10)
    filepath = Path(f"data/test/thermal_spectrum_theta{theta_fmt:03d}.csv")
    fw.write_csv_with_params(df,{"theta":theta},filepath)
    print(f"output {filepath}")

chi = np.logspace(-3,5,256)
compute(theta=0.20,chi=chi)
compute(theta=1.00,chi=chi)
compute(theta=5.00,chi=chi)
