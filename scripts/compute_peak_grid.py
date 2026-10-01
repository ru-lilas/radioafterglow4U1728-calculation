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
        "--model_parameter_table",
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

    conf_lc = GeneralInputs.from_yaml(args.input).model_lightcurves
    table = ThermalSynchrotronIntegrals.from_csv(args.integral_table)
    df_model = FileReader.table_from_csv(args.model_parameter_table)
    model_params = ModelParameters.from_df(df_model)

    results = SynchrotronResults.compute_from(
        model_params=model_params,
        xm=conf_lc.xm.to_floatvector().values[None,:],
        table=table
    )
    peaks = results.peak_for(model_params.synchrotron_model)
    print("dumping now...")
    FileWriter.df_to_csv(
        path=args.output,
        df=peaks.to_df(),
    )
    return

if __name__ == "__main__":
    main()
