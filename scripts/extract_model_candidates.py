import argparse
from pathlib import Path
from module.chevalier import ChevalierGrid
from module.inputs_as_dataclass import GeneralInputs, ModelParameterTable
from module.mydataclasses import QuantityData
from module.observation import LongformatLightcurve
from module.utils import FileWriter

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

    grids = ChevalierGrid.from_csv(args.model_table)
    df_obs_long = LongformatLightcurve.from_csv(args.obslc)

    freqs = [5.5,9.0]
    nu_unit = "GHz"
    phi_unit = "GHz min"
    fnu_unit = "mJy"

    for i,nu in enumerate(freqs):
        df_obs_nu = df_obs_long.extract_lightcurve(
                QuantityData(
                    value = nu,
                    unit = nu_unit
                )
            )
        obs_range = df_obs_nu.peak

        indices_model = grids.allowed_modelindex(
            obs_range,
            phi_unit=phi_unit,
            fnu_unit=fnu_unit
        )
        print(indices_model)

if __name__ == "__main__":
    main()
