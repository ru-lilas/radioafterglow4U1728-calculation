import argparse
from pathlib import Path
from module.fetch_numerical_table import fetch_peak_table
from module.utilities import filereaders as fr
from module.utilities import filewriters as fw
import numpy as np
from module.plot import processor

def main(args:argparse.Namespace):
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    inpath:Path = args.input
    df_input = fr.read_csv(inpath)
    metadata = fr.read_keyvalue(inpath)

    table = fetch_peak_table(args)
    tau_theta_arr = np.asarray(df_input["tau_theta"],dtype=np.float64)

    df_output = df_input.assign(
        xm_peak=table.calculate_xm_peak(tau_theta_arr),
        lambda_peak=table.calculate_lambda_peak(tau_theta_arr),
    )
    phi_peak_arr = processor.calcualte_product_two_columns(
        df=df_output,
        columns=("phi_theta","xm_peak")
    )
    l_peak_arr = processor.calcualte_product_two_columns(
        df=df_output,
        columns=("lnu_theta","lambda_peak")
    )
    df_output = df_output.assign(
        phi_peak=phi_peak_arr,
        lnu_peak=l_peak_arr,
    )
    fw.write_csv_with_params(df_output,metadata,outpath)

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

