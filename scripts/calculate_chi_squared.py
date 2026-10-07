import argparse
from pathlib import Path

import pandas as pd

from module.calculation_lightcurve import BinnedLightcurveWithBackground, ChiSquaredResult
from module.inputs_as_dataclass import GeneralInputs
from module.models import ColumnNames, DefaultUnits, MetadataNames
from module.mydataclasses import LongformatDataFrame, QuantityData
from module.observation import LongformatLightcurve
from module.utils import FileWriter

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("input", "binned_lightcurves", "obslc", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    conf_chi2 = GeneralInputs.from_yaml(args.input).chi_squared
    longdf_binnedlc = LongformatDataFrame.read_from(args.binned_lightcurves)
    binned_lcs = longdf_binnedlc.parse(ColumnNames.NU)
    obslc_long = LongformatLightcurve.from_csv(args.obslc)
    results: list[pd.Series] = []
    nu_unit = DefaultUnits.nu
    for nu_value, longdf_nu in binned_lcs.items():
        if not isinstance(nu_value, float):
            raise TypeError(
                f"nu_valueのtypeが不正です:{type(nu_value)}"
            )
        longdf_nu.attrs[ColumnNames.NU] = nu_value
        nu = QuantityData(
            value=nu_value,
            unit=nu_unit
        )
        timewindow = conf_chi2.timewindow
        obs_lc = obslc_long.select_frequency(nu, unit=nu.unit)
        t_unit = obs_lc.metadata.t_unit
        obs_lc_sampled = obs_lc.select_timewindow(
            t_min=timewindow.t_min.value_in(t_unit),
            t_max=timewindow.t_max.value_in(t_unit)
        )
        dfs_candidate = LongformatDataFrame(longdf_nu).parse(ColumnNames.MODEL_INDEX)

        for model_index, df_cand in dfs_candidate.items():
            df_cand.attrs[ColumnNames.MODEL_INDEX] = model_index
            if not isinstance(model_index, int):
                raise TypeError(
                    f"model_indexのtypeが不正です:{type(model_index)}"
                )
            binned_lc = BinnedLightcurveWithBackground.from_df(
                df_cand,
                obs_lc_sampled.fnu_background.index(0)
            )
            chi2 = ChiSquaredResult.calculate(
                model_index=model_index,
                nu_value=nu_value,
                fnu_model=binned_lc.fnu_with_bg,
                fnu_data=obs_lc_sampled.fnu_total,
                fnu_err=obs_lc_sampled.fnu_total_err,
                n_params=2
            )
            results.append(chi2.to_series())
    df_result = pd.DataFrame(results)
    df_result.attrs[MetadataNames.NU_UNIT] = nu_unit
    FileWriter.df_to_csv(
        args.output,
        df_result,
        save_attrs=True
    )

if __name__ == "__main__":
    main()
