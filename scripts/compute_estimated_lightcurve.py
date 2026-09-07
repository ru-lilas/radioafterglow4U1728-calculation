"""
    compute_estimated_lightcurve.py
    chi-square fittingで推定したパラメータの光度曲線を計算する
"""

import argparse
from pathlib import Path

from module import compute_lightcurve
from module.chevalier import EstimatedParameters
from module.inputs_as_dataclass import GeneralInputs
from module.utils import FileWriter
from module.models import ThermalSynchrotronTable

def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--estimated",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--integral_table",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True
    )
    return parser.parse_args()

def main():
    args = parse_args()
    integral_table = ThermalSynchrotronTable.from_csv(args.integral_table)
    estimated_params = EstimatedParameters.from_csv(args.estimated)
    config = GeneralInputs.from_yaml(args.input)
    config_model = config.chi2fitting.model

    lc = compute_lightcurve.compute(
        config = config_model,
        params = estimated_params,
        integral_table = integral_table,
    )
    print(lc)

    return

if __name__ == "__main__":
    main()
