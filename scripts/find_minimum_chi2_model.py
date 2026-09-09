"""
    find_minimu_chi2_model.py
    許容モデルの中でchi-squaredが最小となるものを見つける
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

    config = GeneralInputs.from_yaml(args.input)
    config_model = config.chi2fitting.model
    config_sampling = config.chi2fitting.sampling

    # calculate lightcurve for each parameter set
    lcs = lc.compute(
        config = config_model,
        params = allowing_params,
        integral_table = integral_table,
    )

    # reduction
    nu_sampling = config_sampling.nu
    timewindow = config_sampling.timewindow
    obslc_long = LongformatLightcurve.from_csv(args.obslc)
    obslc_nu = obslc_long.extract_lightcurve(nu_sampling)
    obslc_nu_selected = obslc_nu.select_timewindow(timewindow)
    binning = obslc_nu_selected.time_bin_bounds(timewindow)
    lc_binned = lcs.bin_average(
        binning,
        drop_incomplete_bin=True
    )

    # calculate chi-squared for bin-averaged lightcurves
    fnu_unit = obslc_nu_selected.fnu_net.unit
    chi2 = CalculateChiSquared.from_grid(
        model = lc_binned.fnu.to_value_in(fnu_unit),
        observed= obslc_nu_selected.fnu_net.to_value_in(fnu_unit),
        error=obslc_nu_selected.fnu_net_err.to_value_in(fnu_unit)
    )
    chi2.to_csv(outpath)

if __name__ == "__main__":
    main()
