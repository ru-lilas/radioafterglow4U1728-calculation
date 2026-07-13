import argparse
from dataclasses import asdict
from pathlib import Path
from module import compute_scaling_parameters
from module.input_reader import read_physical_parameters
from module.utilities import filereaders as fr
from module.utilities import filewriters as fw

def main(args:argparse.Namespace):
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    inpath:Path = args.input
    physical_params = read_physical_parameters(inpath)
    config_data:dict = fr.read_yaml(inpath)

    df = compute_scaling_parameters.table(config_data)
    metadata = {
        **asdict(physical_params.microphysics),
        **asdict(physical_params.distance),
        **asdict(physical_params.units)
    }
    fw.write_csv_with_params(df,metadata,outpath,use_index=True)

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
