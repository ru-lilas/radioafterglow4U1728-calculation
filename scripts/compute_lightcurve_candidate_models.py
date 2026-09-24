"""
    compute_estimated_lightcurve.py
    chi-square fittingで推定したパラメータの光度曲線を計算する
"""

import argparse
from pathlib import Path

from module import mystatistics
from module.inputs_as_dataclass import GeneralInputs
from module.mydataclasses import QuantityData
from module.mystatistics import CalculateChiSquared
from module.observation import LongformatLightcurve
from module.utils import FileReader, FileWriter
from module.synchrotron_numerical_table import DimensionlessThermalLuminosityGrid, ThermalSynchrotronIntegrals
from module.calculation_lightcurve import (
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

    conf_lc = GeneralInputs.from_yaml(args.input).model_lightcurves
    t_array = conf_lc.time.to_QuantityArray1D()
    table = ThermalSynchrotronIntegrals.from_csv(args.integral_table)
    df_models = FileReader.table_from_csv(args.candidate_models)
    models = CandidateModelParameters.from_df(df_models)
    nu_array = models.nu

    phi_grid = PhaseParameterGrid.build(
        t=t_array,
        nu=nu_array
    )
    xm_grid = phi_grid.to_xm(models.phi_th)
    ln_xm_grid = phi_grid.to_ln_xm(models.phi_th)
    log10_xm_grid = phi_grid.to_log10_xm(models.phi_th)
    lambda_th = DimensionlessThermalLuminosityGrid.build(
        tau_th_vector=models.tau_th.decompose(),
        xm_grid=xm_grid,
        table=table
    )
    lnu = SpecificLuminosityGrid.from_FloatGrid(
        grid=lambda_th,
        row_quantity=models.lnu_th
    )
    fnu_src_frame = SourceframeSpecificFluxGrid.from_lnu(
        lnu = lnu,
        distance = models.metadata.distance
    )
    fnu_obs_frame = fnu_src_frame.to_observerframe(
        models.doppler_delta
    )
    lc = CandidateModelLightcurveGrid(
        metadata = models.metadata,
        model_index=models.model_index,
        nu=nu_array,
        t=t_array,
        lnu=lnu,
        fnu_src_frame=fnu_src_frame,
        fnu_obs_frame=fnu_obs_frame
    )
    FileWriter.df_to_csv(
        path=args.output,
        df=lc.to_df(),
    )
    return

if __name__ == "__main__":
    main()
