import argparse
from pathlib import Path

from module.inputs_as_dataclass import GeneralInputs
from module.models import ModelParameters, SynchrotronResults
from module.mydataclasses import QuantityData
from module.utils import FileReader, FileWriter

def parse_args()->argparse.Namespace:
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
    return parser.parse_args()

def main():
    args = parse_args()
    input_model = GeneralInputs.from_yaml(args.input).model_parameters
    models = ModelParameters.build_from_input(input_model)
    print("dumping now...")
    FileWriter.df_to_csv(
        path=args.output,
        df=models.to_df(),
    )

if __name__ == "__main__":
    main()
