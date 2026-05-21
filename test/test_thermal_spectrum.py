
import numpy as np
from pipelines import compute_spectrum

params = {
    "eps_th": 1.0,
    "eps_B": 0.1,
    "mu": 0.62,
    "mu_e": 1.18,
    "beta_sh": 0.1,
    "a_wind_value": 1.0e+07,
    "a_wind_unit": "g/cm",
    "t_value": 3.0,
    "t_unit": "min", 
    "nu_array_value": np.logspace(3,15,512),
    "nu_array_unit": "Hz"
}

compute_spectrum.compute(params)
