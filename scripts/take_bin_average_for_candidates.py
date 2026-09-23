"""
    take_bin_average_for_candidates.py
    candidate models lightcurveのbin平均を計算する
"""

import argparse
from pathlib import Path

from module import compute_lightcurve
from module import mystatistics
from module.inputs_as_dataclass import GeneralInputs
from module.mydataclasses import QuantityData
from module.mystatistics import CalculateChiSquared
from module.observation import LongformatLightcurve
from module.utils import FileReader, FileWriter
from module.synchrotron_numerical_table import DimensionlessThermalLuminosityGrid, ThermalSynchrotronIntegrals
from module.compute_lightcurve import (
    CandidateModelLightcurve,
    CandidateModelLightcurveGrid,
    CandidateModelParameters,
    ObserverframeSpecificFluxGrid,
    PhaseParameterGrid,
    SpecificLuminosityGrid,
    SourceframeSpecificFluxGrid,
    ObserverframeSpecificFluxGrid
    )

def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--candidate_lcs",
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
    conf_chi2 = GeneralInputs.from_yaml(args.input).chi2
    df_candidate_lcs = FileReader.table_from_csv(args.candidate_lcs)
    model_lcs:list[CandidateModelLightcurve] = []
    for cand_idx, df in df_candidate_lcs.groupby("candidate_model_index"):
        if not isinstance(cand_idx,int):
            raise TypeError(
                "indexの型が不正です."
            )
        else:
            model_lcs.append(CandidateModelLightcurve.from_df(cand_idx,df))

    # observation lightcurve data
    obslc_long = LongformatLightcurve.from_csv(args.obslc)
    for cand_lc in model_lcs:
        nu = cand_lc.nu
        obs_lc = obslc_long.select_frequency(nu,unit=nu.unit)
        binning = obs_lc.time_bin_bounds(conf_chi2.timewindow)
        cand_lc_with_zero_boundary = cand_lc.insert_zero_boundary()
        model_binned_lc = cand_lc_with_zero_boundary.bin_average(
            binning,
            drop_incomplete_bin=True
        )
        print(model_binned_lc)
    return

if __name__ == "__main__":
    main()
