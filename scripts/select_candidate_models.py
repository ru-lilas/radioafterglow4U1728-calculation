import argparse
import pandas as pd
from pathlib import Path
from module import chevalier
from module.chevalier import PeakGrid
from module.inputs_as_dataclass import GeneralInputs
from module.mydataclasses import QuantityData
from module.observation import LongformatLightcurve
from module.utils import FileReader, FileWriter
from module.timing_utils import timed

def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--peak_grid",
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

@timed
def main():
    args = parse_args()
    outpath:Path = args.output

    df_grid = FileReader.table_from_csv(args.peak_grid)
    general_conf = GeneralInputs.from_yaml(args.input)
    selection_conf = general_conf.candidate_models_selection
    df_obs_long = LongformatLightcurve.from_csv(args.obslc)
    candidate_longdf = chevalier.build_candidate_longdf(
        df_peaks=df_grid,
        df_obs_long=df_obs_long,
        selection_config=selection_conf
    )

    FileWriter.df_to_csv(
        path=args.output,
        df=candidate_longdf,
        save_attrs=True,
    )

if __name__ == "__main__":
    main()
