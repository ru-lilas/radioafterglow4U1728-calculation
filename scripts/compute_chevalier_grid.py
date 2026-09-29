"""
    compute_chevalier_grid.py
"""

import argparse
from pathlib import Path

from module.inputs_as_dataclass import GeneralInputs
from module.models import ModelParameters, SynchrotronResults
from module.mydataclasses import QuantityData
from module.utils import FileReader, FileWriter
from module.synchrotron_numerical_table import ThermalSynchrotronIntegrals

def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
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

    input_general = GeneralInputs.from_yaml(args.input)
    conf_lc = GeneralInputs.from_yaml(args.input).model_lightcurves
    table = ThermalSynchrotronIntegrals.from_csv(args.integral_table)
    model_params = ModelParameters.build_from_input(input_general.model_parameters)

    results = SynchrotronResults.compute_from(
        model_params=model_params,
        xm=conf_lc.xm.to_floatvector().values[None,:],
        table=table
    )
    peaks = results.peak
    print("dumping now...")
    FileWriter.df_to_csv(
        path=args.output,
        df=peaks.to_df(),
    )
    return

if __name__ == "__main__":
    main()
