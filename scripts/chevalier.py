import argparse
from pathlib import Path

from module.chevalier import (
    ChevalierInputs,
    ChevalierGrid
)

def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--peak_table",
        type=Path,
        required=True
    )
    return parser.parse_args()

def main():
    args = parse_args()
    inputs = ChevalierInputs.import_from(
        path_input=args.input,
        path_peak_table=args.peak_table
    )
    grids = inputs.build_chevalier_grid(
        a_wind_unit="g cm-1",
        phi_unit="GHz min",
        fnu_unit="mJy"
    )
    print(grids)

if __name__ == "__main__":
    main()
