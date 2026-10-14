"""
    compute_candidate.py
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from module.inputs_as_dataclass import GeneralInputs
from module.models import ColumnNames, MetadataNames, ModelParameters, SynchrotronResults
from module.mydataclasses import QuantityData
from module.utils import FileReader, FileWriter
from module.synchrotron_numerical_table import ThermalSynchrotronIntegrals
from module.timing_utils import timed

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
        "--candidate_models",
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

@timed
def main():
    args = parse_args()

    conf_lc = GeneralInputs.from_yaml(args.input).model_lightcurves
    table = ThermalSynchrotronIntegrals.from_csv(args.integral_table)
    df_model = FileReader.table_from_csv(args.model_parameter_table)
    model_params = ModelParameters.from_df(df_model)
    longdf_cand = FileReader.table_from_csv(args.candidate_models)

    dfs_cand = {
        nu: df_cand.drop(columns=ColumnNames.NU)
        for nu,df_cand in longdf_cand.groupby(ColumnNames.NU)
    }

    dfs_lc:list = []

    for nu_value,df_cand in dfs_cand.items():
        if not isinstance(nu_value,float):
            raise TypeError(
                f"nu_valueのtypeが不正です:{type(nu_value)}"
            )
        nu = QuantityData(
            value=nu_value,
            unit=df_cand.attrs[MetadataNames.NU_UNIT]
        )
        indices = list(np.asarray(df_cand[ColumnNames.MODEL_INDEX],dtype=np.int64))
        cand_params = model_params.select_models(indices)

        results = SynchrotronResults.compute_from(
            model_params=cand_params,
            xm=conf_lc.xm.to_floatvector().values[None,:],
            table=table
        )
        results_lc = results.to_lightcurve(nu)
        dfs_lc.append(results_lc.to_df())

    longdf_lc = pd.concat(
        dfs_lc,
        axis="index"
    )
    FileWriter.df_to_csv(
        args.output,
        longdf_lc,
        save_attrs=True
    )
    return

if __name__ == "__main__":
    main()
