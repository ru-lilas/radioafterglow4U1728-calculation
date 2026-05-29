from module import synchrotron_scaling_values
from module.utilities import bisection
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from pathlib import Path

def main(args:argparse.Namespace):
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)
    tau_arr = np.logspace(1,10,512)
    xi_arr = np.empty_like(tau_arr)

    for i,tau in enumerate(tqdm(tau_arr)):
        def f(xi:float):
            return synchrotron_scaling_values.func_ssa_peak(tau,xi)
        xi_arr[i] = bisection.bisection(f,1.0e-1,1.0e+4)

    df = pd.DataFrame({
        "tau_theta":tau_arr,
        "xi_peak":xi_arr
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
