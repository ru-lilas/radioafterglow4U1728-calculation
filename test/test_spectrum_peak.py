import argparse
import numpy as np
from numpy.typing import NDArray
from module import calculate_lnu_th, tabular
from module.utilities import filewriters as fw
from module.utilities import filereaders as fr
from pathlib import Path
import pandas as pd

def fetch_tau_theta_arr(refdata:dict)->NDArray[np.float64]:
    ref_nu_arr = refdata["tau_theta_arr"]
    return np.logspace(**ref_nu_arr,dtype=np.float64)

def fetch_numerical_table(tabular_path:Path):
    df_table = tabular.read_tabular(tabular_path)
    return tabular.ThermalSynchrotronTable(df_table)

def main(args:argparse.Namespace):
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    tabular_path:Path = args.tabular
    inpath:Path = args.input

    refdata:dict = fr.read_yaml(inpath)

    tabular = fetch_numerical_table(tabular_path)
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
        lnu_th_dimless = calculate_lnu_th.dimless_tabular_for_tau_theta(
            x=xm_arr,
            tau_theta=tau_theta,
            table=tabular
        )
    
        idx_peak: int = int(np.argmax(lnu_th_dimless))
        lnu_peak_dimless:NDArray[np.float64] = lnu_th_dimless[idx_peak]
        xm_peak:NDArray[np.float64] = xm_arr[idx_peak]

        outdata.append({
            "tau_theta": tau_theta,
            "xm_peak": xm_peak,
            "lnu_peak_dimless": lnu_peak_dimless
        })

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

