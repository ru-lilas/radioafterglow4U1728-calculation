import argparse
from pathlib import Path
from module.fetch_numerical_table import fetch_tau_table
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

    table = fetch_tau_table(args)
    tau_theta_arr = np.asarray(df_input["tau_theta"],dtype=np.float64)

    df_output = df_input.assign(
        xm_peak=table.calculate_xm_peak(tau_theta_arr),
        lnu_peak_dimless=table.calculate_lnu_peak_dimless(tau_theta_arr),
    )
    phi_peak_arr = processor.calcualte_product_two_columns(
        df=df_output,
        columns=("phi_theta","xm_peak")
    )
    l_peak_arr = processor.calcualte_product_two_columns(
        df=df_output,
        columns=("l_theta","lnu_peak_dimless")
    )
    df_output = df_output.assign(
        phi_peak=phi_peak_arr,
        l_peak=l_peak_arr
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

