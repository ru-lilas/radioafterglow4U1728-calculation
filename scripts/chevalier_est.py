import argparse
from pathlib import Path
from module import plot_utils
from module.chevalier import (
    ChevalierGrid,
    EstimatedParameters,
)
from module.inputs_as_dataclass import PhysicalParameters
from module.observation import LongformatLightcurve
from module.parameter_table import LambdaPeakTable
from module.utils import FileReader, FileWriter

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
    outpath.parent.mkdir(parents=True,exist_ok=True)

    grids = ChevalierGrid.from_yaml(args.input)
    peak_table = LambdaPeakTable.from_csv(args.peak_table)
    df_obs_long = LongformatLightcurve.from_csv(args.obslc)
    df_obs_nu = df_obs_long.extract_lightcurve(9.0)
    obs_scat = df_obs_nu.peak

    phi_unit = "GHz min"
    fnu_unit = "mJy"
    lnu_unit = "erg s-1 Hz-1"

    est_params = grids.select_parameters_in(
        obs_scat,
        peak_table,
        phi_unit=phi_unit,
        fnu_unit=fnu_unit
    )
    df = est_params.to_df(
        phi_unit=phi_unit,
        fnu_unit=fnu_unit,
        lnu_unit=lnu_unit
    )
    FileWriter.df_to_csv(
        path=outpath,
        df=df
    )

if __name__ == "__main__":
    main()
