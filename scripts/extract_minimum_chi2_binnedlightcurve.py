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
    chi2 = mystatistics.ChiSquared.from_csv(args.minchi2)
    minchi2 = chi2.minimum_chi2()

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
    lcs_binned = lcs.bin_average(
        binning,
        drop_incomplete_bin=True
    )
    lc_binned = lcs_binned.extract(
        minchi2=minchi2,
        persistent=obslc_nu_selected.fnu_persistent,
        persistent_err=obslc_nu_selected.fnu_persistent_err,
    )
    df = lc_binned.to_df(
        t_unit = "min",
        a_wind_unit= "g cm-1",
        phi_unit= "GHz min",
        fnu_unit = "mJy",
        lnu_unit = "erg s-1 Hz-1"
    )
    print(df)
    FileWriter.df_to_csv(outpath,df,save_attrs=True)


if __name__ == "__main__":
    main()
