from pathlib import Path
from module.utilities import filereaders as fr
from module.utilities import fetch_file_list
from module.convert_four_vector import convert_beta_into_u
import argparse
import numpy as np
import pandas as pd

def main(args:argparse.Namespace):
    indir:Path = args.datadir
    inpath_list = fetch_file_list.csv(indir)

    df_peak = fetch_df_peak(inpath_list)
    print(df_peak)

    df_peak.to_csv(args.outpath,index=False)
    print(f"output {args.outpath}")


def fetch_df_peak(inpath_list:list[Path]):
    betagamma_sh_arr = []
    beta_sh_arr = []
    nu_peak_arr = []
    lnu_peak_arr = []

    for inpath in inpath_list:
        metadata = fr.read_keyvalue(inpath)
        df = fr.read_csv(inpath)

        beta_sh:float = metadata["beta_sh"]
        betagamma_sh = convert_beta_into_u(beta_sh)

        nu = df["nu"].to_numpy()
        lnu = df["lnu"].to_numpy()
        
        imax: int = int(np.argmax(lnu))
        
        nu_peak: float = float(nu[imax])
        lnu_peak: float = float(lnu[imax])

        betagamma_sh_arr.append(betagamma_sh)
        beta_sh_arr.append(beta_sh)
        nu_peak_arr.append(nu_peak)
        lnu_peak_arr.append(lnu_peak)

    return pd.DataFrame({
        "beta_sh":beta_sh_arr,
        "betagamma_sh":betagamma_sh_arr,
        "nu_peak":nu_peak_arr,
        "lnu_peak":lnu_peak_arr
    })
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "datadir",
        type=Path
    )
    parser.add_argument(
        "outpath",
        type=Path
    )
    
    args = parser.parse_args()
    
    print(f"input {args.datadir}")
    main(args)
