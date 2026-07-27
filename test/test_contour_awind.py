import argparse
from pathlib import Path
from module import compute_contour
from module import build_input_parameters
from module.fetch_numerical_table import fetch_numerical_table
from module.utilities import filereaders as fr
from module.utilities import filewriters as fw

def main(args:argparse.Namespace):
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    inpath:Path = args.input
    inputs:dict = fr.read_yaml_pyyaml(inpath)

    tabular = fetch_numerical_table(args)
    inputparams = build_input_parameters.awind(inputs)
    xm_arr = compute_contour.build_xm_arr(inputs,tabular)
    metadata, df = compute_contour.varying(
        xm_arr=xm_arr,
        tabular=tabular,
        inputparams=inputparams
    )
    fw.write_csv_with_params(df,metadata,outpath)

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
