import argparse
from dataclasses import dataclass
from pathlib import Path
from module.synchrotron_numerical_table import DimensionlessThermalPeak, ThermalSynchrotronIntegrals
from module.mydataclasses import LnSpace, ValueArray, YAMLReadable

def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--synchrotron_function_numerical_table",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True
    )
    return parser.parse_args()

@dataclass(frozen=True,slots=True)
class InputArrays(YAMLReadable):
    ln_tau_th: LnSpace
    ln_xm: LnSpace


def main()->None:
    args=parse_args()

    inputarrs = InputArrays.from_yaml(args.input)
    ip_table = ThermalSynchrotronIntegrals.from_csv(
        args.synchrotron_function_numerical_table
    )

    lambda_table = DimensionlessThermalPeak.build_numerical_table(
        ln_tau_th=inputarrs.ln_tau_th.to_ln(),
        ln_xm=inputarrs.ln_xm.to_ln(),
        integral_table=ip_table
    )

    lambda_table.to_csv(args.output)

    return

if __name__ == "__main__":
    main()


