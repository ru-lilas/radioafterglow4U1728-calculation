from numpy.typing import NDArray
from module import synchrotron_scaling_values
from module.utilities import bisection
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from module import tabular

def main(args:argparse.Namespace):
    tabular_path:Path = args.tabular
    df_table = tabular.read_tabular(tabular_path)
    table = tabular.ThermalSynchrotronTable(df_table)
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)
    tau_theta_arr = np.logspace(1,10,256,dtype=np.float64)
    xi_peak__arr = np.empty_like(tau_theta_arr)

    for i,tau_theta in enumerate(tau_theta_arr):
        def f(xi:NDArray[np.float64]):
            return synchrotron_scaling_values.func_ssa_peak_tablular(xi,tau_theta=tau_theta,table=table)
        xi_peak__arr[i] = bisection.bisection(f,1.0e-1,1.0e+4)

    log_ip_xi_peak = table.calculate_log_ip(xi_peak__arr)
    log_tau_peak = np.log(tau_theta_arr)+log_ip_xi_peak-np.log(xi_peak__arr)
    tau_peak = np.exp(log_tau_peak)
    escape_fraction = -np.expm1(-tau_peak)

    lnu_peak_dimless = synchrotron_scaling_values.calculate_lnu_xi_dimless(
        tau_theta=tau_theta_arr,
        xi=xi_peak__arr
    )

    df = pd.DataFrame({
        "tau_theta":tau_theta_arr,
        "xi_peak":xi_peak__arr,
        "ip_xi_peak":np.exp(log_ip_xi_peak),
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
    parser.add_argument(
        "--tabular",
        type=Path,
        required=True
    )
    args = parser.parse_args()
    main(args)
