import argparse
from pathlib import Path
from typing import Any
from module.utilities import filereaders as fr
from module.utilities import filewriters as fw
from module import arrange_observation_data
import pandas as pd

def main(args:argparse.Namespace):
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    inpath:Path = args.input
    metadata_obs = fr.read_keyvalue(inpath)

    d_value = 5.0
    d_unit = "kpc"
    l_unit = "erg s-1 Hz-1"
    phi_unit = "GHz s"
    phi_peak_5 = arrange_observation_data.calculate_phi_peak(
        nu_value=5.5,
        nu_unit=metadata_obs["nu_unit"],
        t_peak_value=metadata_obs["f5_peak_time"],
        t_unit=metadata_obs["t_unit"],
        phi_unit=phi_unit
    )
    l_peak_5 = arrange_observation_data.convert_fnu_into_lnu(
        fnu_value=metadata_obs["f5_peak_net"],
        fnu_unit=metadata_obs["flux_unit"],
        d_value=d_value,
        d_unit=d_unit,
        l_unit=l_unit
    )
    phi_peak_9 = arrange_observation_data.calculate_phi_peak(
        nu_value=9.0,
        nu_unit=metadata_obs["nu_unit"],
        t_peak_value=metadata_obs["f9_peak_time"],
        t_unit=metadata_obs["t_unit"],
        phi_unit=phi_unit
    )
    l_peak_9 = arrange_observation_data.convert_fnu_into_lnu(
        fnu_value=metadata_obs["f9_peak_net"],
        fnu_unit=metadata_obs["flux_unit"],
        d_value=d_value,
        d_unit=d_unit,
        l_unit=l_unit
    )
    phi_peaks = [phi_peak_5,phi_peak_9]
    l_peaks = [l_peak_5,l_peak_9]
    output_data:dict[str,Any] = {
        "phi_peak": phi_peaks,
        "phi_unit": phi_unit,
        "l_peak": l_peaks,
        "l_unit": l_unit
    }


    metadata = {
        "d_value": d_value,
        "d_unit": d_unit
    }
    df_output = pd.DataFrame(output_data)

    print(df_output)
    fw.write_csv_with_params(df_output,metadata,outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "input",
        type=Path,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True
    )
    args = parser.parse_args()
    main(args)

