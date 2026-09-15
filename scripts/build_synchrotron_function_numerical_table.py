import argparse
from pathlib import Path
from module.synchrotron_numerical_table import ThermalSynchrotronIntegrals
from module.mydataclasses import ValueArray

def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True
    )
    return parser.parse_args()

def main()->None:
    args=parse_args()

    conf = ValueArray.from_yaml(args.input)
    x_arr = conf.arr

    ixm_table = ThermalSynchrotronIntegrals.build_numerical_table(
        xm=x_arr
    )

    ixm_table.to_csv(args.output)

    return

if __name__ == "__main__":
    main()

