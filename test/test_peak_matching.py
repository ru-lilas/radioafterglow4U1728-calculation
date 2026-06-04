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

def fetch_nu_ref(refdata:dict)->float:
    value = float(refdata["nu_ref_value"])
    unit = u.Unit(refdata["nu_ref_unit"])
    quantity = u.Quantity(value,unit)

    return np.atleast_1d(np.asarray(quantity.to_value(u.Unit("Hz")),dtype=np.float64))[0]

def fetch_nu_arr(refdata:dict):
    ref_nu_arr = refdata["nu_arr"]
    return np.logspace(**ref_nu_arr)

def main(args:argparse.Namespace):
    outpath:Path = args.output

    tabular_path:Path = args.tabular
    inpath:Path = args.input

    refdata:dict = fr.read_yaml(inpath)

    df_table = tabular.read_tabular(tabular_path)
    table = tabular.ThermalSynchrotronTable(df_table)
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

    nu_ref = fetch_nu_ref(refdata)
    idx = np.abs(np.log(nu_arr / nu_ref)).argmin()
    t_peak_value_ref = float(t_peak_arr_value[idx])
    print(f"Reference frequency :{nu_ref:.2e} Hz (index i={idx})")
    print(f"Estimated peak time :{t_peak_value_ref:.2e} s")

    lnu_peak_ref_arr = np.asarray(scalings.lnu_peak.to_value(ua.specific_luminosity),dtype=np.float64)
    lnu_peak_ref = lnu_peak_ref_arr[idx]

    spectrum = SynchrotronSpectrum(
        inputs=inputs,
        t_value=t_peak_value_ref,
        t_unit="s",
        nu_value=nu_arr,
        nu_unit="Hz",
        tabular=table
    )

    df = pd.DataFrame({
        "nu":nu_arr,
        "lnu_th":spectrum.lnu_th.to_value(ua.specific_luminosity)
    })

    metadata = {
        "t_peak_ref": t_peak_value_ref,
        "nu_peak_ref": nu_ref,
        "lnu_peak_ref": lnu_peak_ref
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
        "--output",
        type=Path,
        required=True
    )
    args = parser.parse_args()
    main(args)
