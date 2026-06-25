import argparse
from pathlib import Path
from typing import Any
from module.utilities import filewriters as fw
from module import arrange_observation_data
import pandas as pd

def main(args:argparse.Namespace):
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    inpath:Path = args.input
    df_input = pd.read_csv(
        inpath,
        sep=r"\s+",
        header=0,
        comment="#",
        names=[
            "t",
            "f5",
            "f5_err",
            "f9",
            "f9_err"
        ]
    )
    metadata:dict[str,Any] = {
        "t_unit": "min",
        "flux_unit": "mJy",
        "nu_unit": "GHz"
    }
    df_output = df_input.copy()
    df_output["t_err"] = 1.0
    arrange_observation_data.build_arranged_df(metadata,df_output)

    fw.write_csv_with_params(
        df_output,
        metadata,
        outpath
    )
    return 

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

