"""
    compute_estimated_lightcurve.py
    chi-square fittingで推定したパラメータの光度曲線を計算する
"""

import argparse
from pathlib import Path

from module import compute_lightcurve
from module import mystatistics
from module.inputs_as_dataclass import GeneralInputs
from module.mystatistics import CalculateChiSquared
from module.observation import LongformatLightcurve
from module.utils import FileReader, FileWriter
from module.synchrotron_numerical_table import ThermalSynchrotronIntegrals
from module.compute_lightcurve import CandidateModelParameters

def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--candidate_models",
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

    table = ThermalSynchrotronIntegrals.from_csv(args.integral_table)
    df_models = FileReader.table_from_csv(args.candidate_models)
    models = CandidateModelParameters.from_df(df_models)
    print(models)
    # estimated_params = CandidateModelParameters.from_csv(args.estimated)
    # config = GeneralInputs.from_yaml(args.input)
    # config_model = config.chi2fitting.model
    # config_sampling = config.chi2fitting.sampling
    #
    # # calculate lightcurve for each parameter set
    # lcs = compute_lightcurve.compute(
    #     config = config_model,
    #     params = estimated_params,
    #     integral_table = integral_table,
    # )
    #
    # # reduction
    # nu_sampling = config_sampling.nu
    # timewindow = config_sampling.timewindow
    # obslc_long = LongformatLightcurve.from_csv(args.obslc)
    # obslc_nu = obslc_long.extract_lightcurve(nu_sampling)
    # obslc_nu_selected = obslc_nu.select_timewindow(timewindow)
    # binning = obslc_nu_selected.time_bin_bounds(timewindow)
    # lc_binned = lcs.bin_average(
    #     binning,
    #     drop_incomplete_bin=True
    # )
    #
    # # calculate chi-squared for bin-averaged lightcurves
    # fnu_unit = "mJy"
    # chi2_arr = CalculateChiSquared.from_grid(
    #     model = lc_binned.fnu.to_value_in(fnu_unit),
    #     observed= obslc_nu_selected.fnu_net.to_value_in(fnu_unit),
    #     error=obslc_nu_selected.fnu_net_err.to_value_in(fnu_unit)
    # )
    # minimum_chi2 = mystatistics.MinimumChiSquared.from_array(chi2_arr)
    #
    # lc_minimumchi2 = lcs.extract_bestfit_lc(
    #     idx = minimum_chi2.index,
    #     chi2 = minimum_chi2.value
    # )
    # df_ = lc_minimumchi2.to_df()
    # lc_binned_best = lc_binned.extract(
    #     idx=minimum_chi2.index,
    #     chi2=minimum_chi2.value,
    #     persistent=obslc_nu_selected.fnu_persistent,
    #     persistent_err=obslc_nu_selected.fnu_persistent_err,
    # )
    # df = lc_binned_best.to_df(
    #     t_unit = "min",
    #     a_wind_unit= "g cm-1",
    #     phi_unit= "GHz min",
    #     fnu_unit = fnu_unit,
    #     lnu_unit = "erg s-1 Hz-1"
    # )
    # FileWriter.df_to_csv(outpath,df,save_attrs=True)
    #
    return

if __name__ == "__main__":
    main()
