from module import synchrotron_function, synchrotron_scaling_values
from module.utilities import bisection
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from pathlib import Path

def main(args:argparse.Namespace):
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)
    tau_theta_arr = np.logspace(1,10,256,dtype=np.float64)
    xi_arr = np.empty_like(tau_theta_arr)

    for i,tau_theta in enumerate(tqdm(tau_theta_arr)):
        def f(xi:float):
            return synchrotron_scaling_values.func_ssa_peak(tau_theta,xi)
        xi_arr[i] = bisection.bisection(f,1.0e-1,1.0e+4)

    ip_xi_peak = synchrotron_function.thermal_Ip(xi_arr)
    log_tau_peak = np.log(tau_theta_arr)+np.log(ip_xi_peak)-np.log(xi_arr)
    tau_peak = np.exp(log_tau_peak)
    escape_fraction = -np.expm1(-tau_peak)

    lnu_peak_dimless = synchrotron_scaling_values.calculate_lnu_xi_dimless(
        tau_theta=tau_theta_arr,
        xi=xi_arr
    )

    df = pd.DataFrame({
        "tau_theta":tau_theta_arr,
        "xi_peak":xi_arr,
        "ip_xi_peak":ip_xi_peak,
        "tau_peak":tau_peak,
        "f_esc":escape_fraction,
        "lnu_peak_dimless":lnu_peak_dimless,
    })

    df.to_csv(outpath,index=False)
    print(f"output {outpath}")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "output",
        type=Path,
    )
    args = parser.parse_args()
    main(args)
