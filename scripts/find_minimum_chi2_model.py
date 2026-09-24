"""
    find_minimu_chi2_model.py
    許容モデルの中でchi-squaredが最小となるものを見つける
"""

import argparse
from pathlib import Path

from module import calculation_lightcurve as lc
from module import mystatistics
from module.inputs_as_dataclass import GeneralInputs
from module.mydataclasses import WideFormatDataFrames
from module.mystatistics import CalculateChiSquared
from module.observation import LongformatLightcurve
from module.utils import FileReader, FileWriter
from module.models import ThermalSynchrotronTable

def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--candidate_binned_lightcurves",
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

    df_long = FileReader.table_from_csv(args.candidate_binned_lightcurves)

    candidate_lc_table = lc.CandidateModelLightcurveTable.from_df_long(df_long)
    candidate_lcs_nu = candidate_lc_table.devide_into_frequency_groups()
    dfs = [
        lc_nu.to_df_minimum_chi2()
        for lc_nu in candidate_lcs_nu
    ]
    dfs_wide = WideFormatDataFrames(dfs)
    df_long_minchi2 = dfs_wide.to_long_format()

    FileWriter.df_to_csv(
        path=args.output,
        df=df_long_minchi2
    )

if __name__ == "__main__":
    main()
