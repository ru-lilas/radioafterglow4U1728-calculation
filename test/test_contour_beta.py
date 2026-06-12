import argparse
import numpy as np
from numpy.typing import NDArray
from module.find_peak_properties import build_peak_property
from module.models import InputParameters
from module.utilities import filewriters as fw
from module.utilities import filereaders as fr
from module.fetch_numerical_table import fetch_numerical_table
from pathlib import Path
import pandas as pd

def build_input_parameters(inputs:dict):
    beta_arr:NDArray[np.float64] = np.linspace(**inputs["beta_arr"],dtype=np.float64)
    inputparams:list[InputParameters] = []
    for beta in beta_arr:
        inputparams.append(InputParameters(
            **inputs["fixed"],
            beta_sh=beta
        ))
    return inputparams


def main(args:argparse.Namespace):
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    inpath:Path = args.input
    inputs:dict = fr.read_yaml(inpath)

    tabular = fetch_numerical_table(args)
    inputparams = build_input_parameters(inputs)
    # tau_theta_arr = fetch_tau_theta_arr(inputs)


    xm_num:int = inputs["xm_num"]
    xm_arr = np.logspace(
        start=np.log(tabular.xi_min),
        stop=np.log(tabular.xi_max),
        num=xm_num,
        dtype=np.float64
    )
    outdata:list[dict] = []
    for inputparam in inputparams:
        tau_theta = inputparam.tau_theta
        peak_data = build_peak_property(xm_arr,tau_theta,tabular)
        outdata.append({
            **peak_data
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

