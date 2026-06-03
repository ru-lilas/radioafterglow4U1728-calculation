from module import synchrotron_function, synchrotron_scaling_values
from module.utilities import bisection
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from module.models import SynchrotronScalingValues,InputParameters
from pathlib import Path
import astropy.units as u

def main(args:argparse.Namespace):
    # outpath:Path = args.output
    # outpath.parent.mkdir(parents=True,exist_ok=True)

    inputs = InputParameters(
        eps_th=1.0,
        eps_B=0.1,
        mu=0.62,
        mu_e=1.18,
        beta_sh=0.1,
        a_wind_value=1.0e+07,
        a_wind_unit="g/cm"
    )
    values = SynchrotronScalingValues(
        input=inputs,
        nu_value=9.0,
        nu_unit="GHz"
    )

    tau_theta = values.tau_theta[0]
    t_peak = values.t_peak.to(u.Unit("s"))
    print(f"nu = {values.nu:.2e}")
    print(f"tau_theta = {tau_theta:.2e}")
    print(f"phi_theta = {values.phi_theta:.2e}")
    print(f"xi_peak = {values.xi_peak:.2e}")
    print(f"phi_peak = {values.phi_peak:.2e}")
    print(f"t_peak = {t_peak:.2e}")
    print(f"lnu_peak = {values.lnu_peak:.2e}")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # parser.add_argument(
    #     "output",
    #     type=Path,
    # )
    args = parser.parse_args()
    main(args)
