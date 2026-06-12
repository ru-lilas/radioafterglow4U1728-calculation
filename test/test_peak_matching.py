import argparse
import numpy as np
from module.models import ThermalSynchrotronScalingValues,InputParameters,SynchrotronSpectrum
from module import tabular
from module.utilities import filewriters as fw
from module.utilities import filereaders as fr
from module.utilities import unit_aliases as ua
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
    t_peak_arr = scalings.t_peak
    t_peak_arr_value = np.asarray(t_peak_arr.to_value(u.Unit("s")),dtype=np.float64)
    t_unit = str(t_peak_arr.unit)
    t_peak_list:list[float] = list(t_peak_arr_value)

    lnu_peak_ref_arr = np.asarray(scalings.lnu_peak.to_value(ua.specific_luminosity),dtype=np.float64)
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
            "lnu_th":spectrum.lnu_th.to_value(ua.specific_luminosity),
            "ln_tau":ln_tau,
            "tau": tau,
            "exp(-tau)": np.exp(-tau),
            "f_esc": -np.expm1(-tau),
            "xi_f_esc": spectrum.xi*(-np.expm1(-tau)),
        })

        lnu_ref = lnu_peak_ref_arr[i]
        l_theta = scalings.l_theta.value[i]
        l_unit = scalings.l_theta.unit
        metadata = {
            "t": t_ref,
            "t_unit": t_unit,
            "nu_peak_ref": nu_arr[i],
            "lnu_peak_ref": lnu_ref,
            "l_theta": l_theta,
            "l_unit": l_unit,
            "lnu_ref/l_theta": lnu_ref/l_theta,
            "xi_peak": scalings.xi_peak,
            "phi_theta": inputs.phi_theta.value,
            "phi_unit": inputs.phi_theta.unit
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
