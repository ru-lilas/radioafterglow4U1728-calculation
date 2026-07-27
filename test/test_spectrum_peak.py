import argparse
import numpy as np
from numpy.typing import NDArray
from module import calculate_lnu_th
from module.utilities import filewriters as fw
from module.utilities import filereaders as fr
from module.fetch_numerical_table import fetch_numerical_table
from module.find_peak_properties import build_peak_property
from pathlib import Path
import pandas as pd

def fetch_tau_theta_arr(refdata:dict)->NDArray[np.float64]:
    ref_nu_arr = refdata["tau_theta_arr"]
    return np.logspace(**ref_nu_arr,dtype=np.float64)

def main(args:argparse.Namespace):
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    inpath:Path = args.input

    refdata:dict = fr.read_yaml_pyyaml(inpath)

    tabular = fetch_numerical_table(args)
    tau_theta_arr = fetch_tau_theta_arr(refdata)
    xi_num:int = refdata["xi_num"]
    xm_arr = np.logspace(
        start=np.log(tabular.xi_min),
        stop=np.log(tabular.xi_max),
        num=xi_num,
        dtype=np.float64
    )

    outdata:list[dict] = []
    for tau_theta in tau_theta_arr:
        outdata.append(
            build_peak_property(xm_arr,tau_theta,tabular)
        )

    df = pd.DataFrame(outdata)
    fw.write_csv_with_params(df,{},outpath)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "input",
        type=Path,
    )
    parser.add_argument(
        "--tabular",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True
    )
    args = parser.parse_args()
    main(args)

