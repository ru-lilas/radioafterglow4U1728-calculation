"""
    compute_estimated_lightcurve.py
    chi-square fittingで推定したパラメータの光度曲線を計算する
"""

import argparse
from pathlib import Path

from module.chi2_fitting import MinimumChi2Summary
from module import compute_lightcurve,observation
from module.inputs_as_dataclass import GeneralInputs
from module.utils import FileWriter
from module.models import ThermalSynchrotronTable

def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--estimated",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--parameter_table",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--integral_table",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--obs_lc",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--config",
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

    chi2min_est = MinimumChi2Summary.from_csv(args.estimated)
    idx_est:int = chi2min_est.idx

    df_param = read_as_df(args.parameter_table)

    input_est = compute_lightcurve.Input.from_idx(
        df = df_param,
        idx = idx_est
    )

    conf = GeneralInputs.from_yaml(args.config)
    conf_fitting = conf.chi2fitting
    conf_sampling = conf_fitting.sampling

    obslc_general = observation.LongformatLightcurve.from_csv(args.obs_lc)
    obslc_nu = obslc_general.extract_lightcurve(conf_sampling.nu.value)
    obslc_nu_selected = obslc_nu.select_timewindow(conf_sampling.timewindow)
    binning = obslc_nu_selected.time_bin_bounds(conf_sampling.timewindow)

    lc_conf = conf_fitting.model
    table_integral = ThermalSynchrotronTable.from_csv(
        path = args.integral_table
    )
    lc_model = compute_lightcurve.ThermalSynchrotronUtils(table_integral)

    lc = compute_lightcurve.compute(
        config=lc_conf,
        model=lc_model,
        input=input_est
    )
    lc_binned = lc.bin_average(
        binning=binning,
        drop_incomplete_bin=True
    )
    lc_predicted = lc_binned.add_persistent_flux(
        persistent=obslc_nu_selected.fnu_persistent,
        persistent_err=obslc_nu_selected.fnu_persistent_err
    )

    df = lc_predicted.to_df(t_unit="min",fnu_unit="mJy")
    FileWriter.df_to_csv(
        args.output,
        df,
        save_attrs=True,
        save_index=False
    )
    return

if __name__ == "__main__":
    main()
