"""
    extract_minimum_chi2_lightcurve.py
    chi-squaredが最小であるモデルの光度曲線データを出力する
"""

import argparse
from pathlib import Path

from module import compute_lightcurve as lc
from module import mystatistics
from module.inputs_as_dataclass import GeneralInputs
from module.mystatistics import CalculateChiSquared
from module.observation import LongformatLightcurve
from module.utils import FileWriter
from module.models import ThermalSynchrotronTable

def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--minchi2",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--allowing",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--integral_table",
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

    integral_table = ThermalSynchrotronTable.from_csv(args.integral_table)
    allowing_params = lc.AllowningParameters.from_csv(args.allowing)
    minchi2 = mystatistics.MinimumChiSquared.from_yaml(args.minchi2)

    allowing_params = lc.AllowningParameters.from_csv(args.allowing)

    config = GeneralInputs.from_yaml(args.input)
    config_model = config.chi2fitting.model
    config_sampling = config.chi2fitting.sampling

    # calculate lightcurve for each parameter set
    lcs = lc.compute(
        config = config_model,
        params = allowing_params,
        integral_table = integral_table,
    )
    lc_minchi2 = lcs.extract_bestfit_lc(minchi2)
    lc_minchi2.dump(outpath)

if __name__ == "__main__":
    main()
