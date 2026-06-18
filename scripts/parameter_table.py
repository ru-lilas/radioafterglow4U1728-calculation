import argparse
from pathlib import Path
from module import build_input_parameters
from module.utilities import filereaders as fr
from module.utilities import filewriters as fw

def main(args:argparse.Namespace):
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    inpath:Path = args.input
    inputs:dict = fr.read_yaml(inpath)

    df = build_input_parameters.table(inputs)
    metadata = inputs["fixed"]
    fw.write_csv_with_params(df,metadata,outpath)

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
