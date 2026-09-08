"""
    compute_estimated_lightcurve.py
    chi-square fittingで推定したパラメータの光度曲線を計算する
"""

import argparse
from pathlib import Path

from module import compute_lightcurve
from module import mystatistics
from module.chevalier import EstimatedParameters
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
        "--estimated",
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
    integral_table = ThermalSynchrotronTable.from_csv(args.integral_table)
    estimated_params = EstimatedParameters.from_csv(args.estimated)
    config = GeneralInputs.from_yaml(args.input)
    config_model = config.chi2fitting.model
    config_sampling = config.chi2fitting.sampling

    lc = compute_lightcurve.compute(
        config = config_model,
        params = estimated_params,
        integral_table = integral_table,
    )

    nu_sampling = config_sampling.nu
    timewindow = config_sampling.timewindow
    obslc_long = LongformatLightcurve.from_csv(args.obslc)
    obslc_nu = obslc_long.extract_lightcurve(nu_sampling)
    obslc_nu_selected = obslc_nu.select_timewindow(timewindow)
    binning = obslc_nu_selected.time_bin_bounds(timewindow)
    lc_binned = lc.bin_average(
        binning,
        drop_incomplete_bin=True
    )
    fnu_unit = "mJy"
    chi2_arr = CalculateChiSquared.from_grid(
        model = lc_binned.fnu.to_value_in(fnu_unit),
        observed= obslc_nu_selected.fnu_net.to_value_in(fnu_unit),
        error=obslc_nu_selected.fnu_net_err.to_value_in(fnu_unit)
    )
    minimum_chi2 = mystatistics.MinumumChiSquared.from_array(chi2_arr)
    print(minimum_chi2)

    return

if __name__ == "__main__":
    main()
