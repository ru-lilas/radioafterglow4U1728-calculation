"""
    spectrum_peak_evolution.py
"""
from pathlib import Path
from module.utilities import filereaders as fr
import argparse
import pandas as pd
from module.extract_spectrum_peak import extract_peak_row

def build_spectrum_peak_evolution(inpath_list:list[Path]):
    output_data = []
    for inpath in inpath_list:
        metadata = fr.read_keyvalue(inpath)
        df = fr.read_csv(inpath)
        spectrum_peak = extract_peak_row(df)
        output_data.append({
            **metadata,
            **spectrum_peak
        })

    return pd.DataFrame(output_data)

def main(args:argparse.Namespace):

    outpath:Path = args.output
    inpath_list:list[Path] = args.input

    output_df = build_spectrum_peak_evolution(inpath_list)
    output_df.to_csv(outpath,index=False)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "input",
        type=Path,
        nargs='*'
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True
    )
    
    args = parser.parse_args()
    
    main(args)

