"""
    build_lightcurve.py
"""
# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false
import astropy.units as u
from pathlib import Path
from module.dump_csv import dump_csv
from module.utilities import filereaders as fr
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd

def parse_lightcurve_config(config:dict):
    unit = u.Unit(config["unit"])
    values = np.asarray(config["values"])
    quantity = u.Quantity(values,unit)
    return np.asarray(quantity.to_value(u.Hz))

def fetch_(df:pd.DataFrame,band_center:float):
    row = df.iloc[(df["nu"] - band_center).abs().argmin()]
    return row

def build_lc_row(metadata:dict,df:pd.DataFrame,nu:float):
    t = float(metadata["t_value"])
    row = fetch_(df,nu)
    return float(row["nu"]),{
        "t":t,
        "lnu": float(row["lnu"]),
    }

def main(args:argparse.Namespace):

    outpath:Path = args.output
    inpath_list:list[Path] = args.input
    config = fr.read_yaml(args.config)
    nu_refs = parse_lightcurve_config(config)
    lc_longformat = []

    # スペクトルの時間発展データを取得
    for inpath in inpath_list:
        metadata = fr.read_keyvalue(inpath)
        df = fr.read_csv(inpath)
        for nu_ref in nu_refs:
            nu,lc_row = build_lc_row(metadata,df,nu_ref)
            lc_single_line = {
                "nu": nu,
                **metadata,
                **lc_row
            }
            lc_longformat.append(lc_single_line)

    df = pd.DataFrame(lc_longformat)
    df.to_csv(outpath,index=False)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "input",
        type=Path,
        nargs='*'
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True
    )
    
    args = parser.parse_args()
    
    main(args)
