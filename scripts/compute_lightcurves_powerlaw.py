import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd

from module.inputs_as_dataclass import GeneralInputs
from module.models import ColumnNames, MetadataNames, ModelParameters, SwitchKeys, SynchrotronResults
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
        "--model_parameter_table_powerlaw",
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
    df_powerlaw = FileReader.table_from_csv(args.model_parameter_table_powerlaw)
    thermal_models = ModelParameters.from_df(df_model)

    dfs_lc:list = []
    for row_pl in df_powerlaw.to_dict(orient="records"):
        thermal_model_index= int(row_pl[MetadataNames.THERMAL_MODEL_INDEX])
        model_index = int(row_pl[ColumnNames.POWERLAW_SUBMODEL_INDEX])
        p_index = row_pl[MetadataNames.P_INDEX]
        f_e = row_pl[MetadataNames.F_E]

        nu = QuantityData(
            value=row_pl[ColumnNames.NU],
            unit=df_powerlaw.attrs[MetadataNames.NU_UNIT]
        )

        thermal_model = thermal_models.select_models([thermal_model_index])

        powerlaw_model = replace(
            thermal_model,
            powerlaw = replace(
                thermal_model.powerlaw,
                p_index=p_index,
                f_e = f_e,
            ),
            switches = {
                **thermal_model.switches,
                SwitchKeys.THERMAL_ONLY: False
            },
            model_index=np.full_like(
                thermal_model.model_indices,
                model_index,
            ),
        )
        results = SynchrotronResults.compute_from(
            model_params=powerlaw_model,
            xm=conf_lc.xm.to_floatvector().values[None,:],
            table=table
        )
        results_lc = results.to_lightcurve(nu)
        df_lc = results_lc.to_df(
            index_column_name=ColumnNames.POWERLAW_SUBMODEL_INDEX
        )
        df_lc[MetadataNames.THERMAL_MODEL_INDEX] = thermal_model_index
        dfs_lc.append(df_lc)
    longdf_lc = pd.concat(
        dfs_lc,
        axis="index"
    )
    # 同じIDの組には、時刻・周波数によらず同じmodel_indexを付ける
    model_keys = pd.MultiIndex.from_frame(longdf_lc.loc[:, [
        ColumnNames.POWERLAW_SUBMODEL_INDEX,
        MetadataNames.THERMAL_MODEL_INDEX,
    ]])
    model_indices, _ = model_keys.factorize(sort=True)
    longdf_lc.insert(0, ColumnNames.MODEL_INDEX, model_indices)
    FileWriter.df_to_csv(
        path=args.output,
        df=longdf_lc,
        save_attrs=True
    )
    return

if __name__ == "__main__":
    main()
