import argparse
from pathlib import Path
from module.inputs_as_dataclass import GeneralInputs
from module.utils import FileWriter

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
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    general_inputs = GeneralInputs.from_yaml(args.input)
    model_params = general_inputs.physical_parameters
    model_table = model_params.build()

    df = model_table.to_df()
    FileWriter.df_to_csv(
        path=args.output,
        df = df,
        save_attrs= True,
        save_index= True
    )

if __name__ == "__main__":
    main()
