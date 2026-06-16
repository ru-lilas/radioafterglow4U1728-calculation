import argparse
from pathlib import Path
from module import compute_contour
from module.fetch_numerical_table import fetch_numerical_table
from module.utilities import filereaders as fr
from module.utilities import filewriters as fw
import pandas as pd

def main(args:argparse.Namespace):
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    inpath:Path = args.input
    inputs:dict = fr.read_yaml(inpath)

    tabular = fetch_numerical_table(args)
    xm_arr = compute_contour.build_xm_arr(inputs,tabular)
    tau_theta_arr = compute_contour.build_tau_theta_arr(inputs)
    outdata = compute_contour.peak_table(
        xm_arr=xm_arr,
        tau_theta_arr=tau_theta_arr,
        integral_table=tabular,
    )
    df = pd.DataFrame(outdata)
    fw.write_csv_with_params(df,{},outpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "input",
        type=Path,
    )
    parser.add_argument(
        "--table",
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

