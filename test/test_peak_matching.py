import argparse
from typing import cast
import numpy as np
from module.models import ThermalSynchrotronScalingValues,InputParameters,SynchrotronSpectrum
from module import tabular
from module.utilities import filewriters as fw
from module.utilities import filereaders as fr
from pathlib import Path
import pandas as pd
import astropy.units as u

def fetch_nu_arr(refdata:dict):
    ref_nu_arr = refdata["nu_arr"]
    return np.logspace(**ref_nu_arr)

def fetch_numerical_table(tabular_path:Path):
    df_table = tabular.read_tabular(tabular_path)
    return tabular.ThermalSynchrotronTable(df_table)

def main(args:argparse.Namespace):
    outdir:Path = args.outdir
    outdir.mkdir(parents=True,exist_ok=True)

    tabular_path:Path = args.tabular
    inpath:Path = args.input

    refdata:dict = fr.read_yaml(inpath)

    table = fetch_numerical_table(tabular_path)
    nu_arr = fetch_nu_arr(refdata)

    inputs = InputParameters(**refdata["inputs"])
    scalings = ThermalSynchrotronScalingValues(
        input=inputs,
        nu_value=nu_arr,
        nu_unit="Hz",
        table=table
    )
    xm_est = scalings.xi_est
    t_peak_arr = scalings.t_peak
    t_peak_arr_value = np.asarray(t_peak_arr.to_value(u.Unit("s")),dtype=np.float64)
    t_unit = str(t_peak_arr.unit)
    t_peak_list:list[float] = list(t_peak_arr_value)

    for i,t_ref in enumerate(t_peak_list):
        outpath = outdir/f"{i:03d}.csv"

        spectrum = SynchrotronSpectrum(
            inputs=inputs,
            t_value=t_ref,
            t_unit="s",
            nu_value=nu_arr,
            nu_unit="Hz",
            tabular=table
        )

        ln_tau = spectrum.ln_tau
        tau = np.exp(ln_tau)
        df = pd.DataFrame({
            "xi":spectrum.xi,
            "nu":nu_arr,
            "lnu_th_dimless":spectrum.lnu_th_dimless,
            "ln_tau":ln_tau,
            "f_esc": -np.expm1(-tau),
        })

        lnu_est = scalings.lnu_est_dimless
        l_theta = inputs.l_theta.value
        l_unit = cast(u.UnitBase,inputs.l_theta.unit)

        # peak luminosity
        row_peak = pd.DataFrame(df.loc[[df["lnu_th_dimless"].idxmax()]])
        xi_peak = float(row_peak.iloc[0]["xi"])
        lnu_peak = float(row_peak.iloc[0]["lnu_th_dimless"])

        metadata = {
            "t": t_ref,
            "t_unit": t_unit,
            "xm_est": xm_est,
            "lnu_est_dimless": lnu_est,
            "l_theta": l_theta,
            "l_unit": l_unit.to_string(format="fits"),
            "phi_theta": inputs.phi_theta.value,
            "phi_unit": inputs.phi_theta.unit,
            "lnu_peak_dimless": lnu_peak,
            "xm_peak": xi_peak,
        }

        fw.write_csv_with_params(df,metadata,outpath)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "input",
        type=Path,
    )
    parser.add_argument(
        "--tabular",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True
    )
    args = parser.parse_args()
    main(args)
