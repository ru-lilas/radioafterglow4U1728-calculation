import argparse
from pathlib import Path
from module.utilities import filereaders as fr
from module.utilities import filewriters as fw
import pandas as pd

def main(args:argparse.Namespace):
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    inpath:Path = args.input
    df_input = fr.read_csv(inpath)
    metadata = fr.read_keyvalue(inpath)

    columns:list[str] = [
        "beta_sh",
        "a_wind_value",
        "phi_peak",
        "phi_unit",
        "l_peak",
        "l_unit",
    ]

    df_output = pd.DataFrame(df_input[columns])


    fw.write_csv_with_params(df_output,metadata,outpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "input",
        type=Path,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True
    )
    args = parser.parse_args()
    main(args)
