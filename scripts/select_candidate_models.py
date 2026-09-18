import argparse
import pandas as pd
from pathlib import Path
from module.chevalier import ChevalierGrid
from module.inputs_as_dataclass import GeneralInputs, ModelParameterTable
from module.mydataclasses import QuantityData
from module.observation import LongformatLightcurve
from module.utils import FileReader, FileWriter

def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--model_table",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--obslc",
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
    outpath:Path = args.output

    general_conf = GeneralInputs.from_yaml(args.input)
    selection_conf = general_conf.candidate_models_selection
    units = general_conf.units

    df_model = FileReader.table_from_csv(
        path=args.model_table,
        idx="index"
    )
    grids = ChevalierGrid.from_df(df_model)
    df_obs_long = LongformatLightcurve.from_csv(args.obslc)

    obs_freqs = selection_conf.observation_frequencies.to_QuantityData_list()
    dfs_obs_nu = [
        df_obs_long.extract_lightcurve(nu)
        for nu in obs_freqs
    ]

    dfs_candidates:list = []
    for i,df_obs_nu in enumerate(dfs_obs_nu):
        obs_range = df_obs_nu.peak

        dfs_candidates.append(
            grids.df_modelcandidates(
                df=df_model,
                obs=obs_range,
                phi_unit=units.phi_unit,
                fnu_unit=units.fnu_unit
            )
        )

    df_long = pd.concat(
        dfs_candidates,
        axis="index"
    ).rename_axis("model_index").reset_index()

    FileWriter.df_to_csv(
        path=args.output,
        df=df_long,
        save_attrs=True
    )

if __name__ == "__main__":
    main()
