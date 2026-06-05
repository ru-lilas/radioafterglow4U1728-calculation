import argparse
import numpy as np
from module.utilities import filewriters as fw
from module.utilities import filereaders as fr
from pathlib import Path
import pandas as pd

xname = "nu"
yname = "lnu_th"

def main(args:argparse.Namespace):
    inpaths:list[Path] = args.input
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    outdata_list:list[dict] = []

    for inpath in inpaths:
        df_in = fr.read_csv(inpath)
        metadata = fr.read_keyvalue(inpath)

        t_value_ref = metadata["t_peak_ref"]

        nu_ref = metadata["nu_peak_ref"]
        lnu_peak_estimated = metadata["lnu_peak_ref"]

        # peak luminosity
        row_peak = pd.DataFrame(df_in.loc[[df_in[yname].idxmax()]])
        nu_peak = float(row_peak.iloc[0][xname])
        lnu_peak = float(row_peak.iloc[0][yname])
        nu_err = 1.0 - nu_peak/nu_ref
        lnu_err = 1.0 - lnu_peak/lnu_peak_estimated

        outdata_list.append({
            "t_ref": t_value_ref,
            "nu_ref": nu_ref,
            "lnu_est": lnu_peak_estimated,
            "nu_peak": nu_peak,
            "lnu_peak": lnu_peak,
            "nu_err": nu_err,
            "lnu_err": lnu_err
        })

    df = pd.DataFrame(outdata_list)
    fw.write_csv_with_params(df,{},outpath)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "input",
        type=Path,
        nargs="*"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True
    )
    args = parser.parse_args()
    main(args)
