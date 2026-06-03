from argparse import Namespace
import argparse
from pathlib import Path
from numpy.typing import NDArray
from module import synchrotron_function
from module.utilities import filereaders as fr
import numpy as np
import pandas as pd

def calculate__ipxi_devided_xi2(xi:NDArray[np.float64]):
    ipxi = synchrotron_function.thermal_Ip(xi)
    return ipxi/xi**2

def calculate__ipxi_devided_xi(xi:NDArray[np.float64]):
    ipxi = synchrotron_function.thermal_Ip(xi)
    return ipxi/xi

def main(args:Namespace):
    inpath:Path = args.input
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    xi_input = fr.read_yaml(inpath)
    xi_array = np.logspace(
        start = float(xi_input["log10_min"]),
        stop = float(xi_input["log10_max"]),
        num = int(xi_input["num"])
    )
    ipxi = synchrotron_function.thermal_Ip(xi_array)

    df = pd.DataFrame({
        "xi":xi_array,
        "ip":ipxi,
        "ip_xi_inv":ipxi/xi_array,
        "ip_xi2_inv":ipxi/xi_array**2
    })

    df.to_csv(outpath,index=False)
    print(f"output {outpath}")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "input",
        type=Path
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    main(args)

