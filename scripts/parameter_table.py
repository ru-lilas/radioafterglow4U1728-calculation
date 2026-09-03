import argparse
from dataclasses import asdict
from pathlib import Path

from dacite import from_dict
from module.input_reader import read_physical_parameters
from module.parameter_table import PhysicalParameters:
from module.utils import FileReader
from module.utilities import filewriters as fw

def main(args:argparse.Namespace):
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    inputs:dict = FileReader.yaml_safe(args.input)
    physical_params = from_dict(
        data_class=PhysicalParameters:,
        data = inputs["physical_parameters"]
    )

    df = physical_params.to_df()
    metadata = physical_params.metadata()
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
