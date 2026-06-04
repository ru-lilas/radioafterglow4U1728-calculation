import argparse
import numpy as np
from module.models import ThermalSynchrotronScalingValues,InputParameters,SynchrotronSpectrum
from module import tabular
from module.utilities import filewriters as fw
from module.utilities import unit_aliases as ua
from pathlib import Path
import pandas as pd
import astropy.units as u

def main(args:argparse.Namespace):
    outpath:Path = args.output

    tabular_path:Path = args.tabular
    df_table = tabular.read_tabular(tabular_path)
    table = tabular.ThermalSynchrotronTable(df_table)
    nu_arr =np.logspace(
            start=6.0,
            stop=12.0,
            num=256
        )

    inputs = InputParameters(
        eps_th=1.0,
        eps_B=0.1,
        mu=0.62,
        mu_e=1.18,
        beta_sh=0.1,
        a_wind_value=1.0e+07,
        a_wind_unit="g/cm"
    )
    scalings = ThermalSynchrotronScalingValues(
        input=inputs,
        nu_value=nu_arr,
        nu_unit="Hz",
        table=table
    )
    t_peak_arr = scalings.t_peak
    t_peak_arr_value = np.asarray(t_peak_arr.to_value(u.Unit("s")),dtype=np.float64)

    i = 160
    print(f"Reference index i={i}")
    nu_ref = nu_arr[i]
    lnu_peak_ref_arr = np.asarray(scalings.lnu_peak.to_value(ua.specific_luminosity),dtype=np.float64)
    lnu_peak_ref = lnu_peak_ref_arr[i]
    t_peak_value_ref:float = t_peak_arr_value[i]

    spectrum = SynchrotronSpectrum(
        inputs=inputs,
        t_value=t_peak_value_ref,
        t_unit="s",
        nu_value=nu_arr,
        nu_unit="Hz",
        tabular=table
    )

    metadata = {
        "t_peak_ref": t_peak_value_ref,
        "nu_peak_ref": nu_ref,
        "lnu_peak_ref": lnu_peak_ref
    }
    df = pd.DataFrame({
        "nu":nu_arr,
        "lnu_th":spectrum.lnu_th.to_value(ua.specific_luminosity)
    })

    fw.write_csv_with_params(df,metadata,outpath)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--tabular",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True
    )
    args = parser.parse_args()
    main(args)
