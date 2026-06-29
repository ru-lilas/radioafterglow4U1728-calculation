import argparse
from pathlib import Path
from typing import cast
from module import dataframe_processors
from module.utilities import filereaders as fr
from module.utilities import filewriters as fw
import pandas as pd
import numpy as np
from module.utilities.quantity_data import QuantityData
from module import quantity_converter
import astropy.units as u

def main(args:argparse.Namespace):
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    confpath:Path = args.config
    conf = fr.read_yaml(confpath)
    dst_xrb = QuantityData(
        value = np.asarray(conf["d_value"],dtype=np.float64),
        unit = conf["d_unit"]
    )

    inpath:Path = args.input
    df_input = fr.read_csv(inpath)
    metadata_input = fr.read_keyvalue(inpath)

    lnu_peak_value_arr = dataframe_processors.convert_ndarray(
        df_input,
        "lnu_peak"
    )
    lnu_peak = QuantityData(
        value = lnu_peak_value_arr,
        unit = metadata_input["lnu_unit"]
    )
    fnu_peak_q = quantity_converter.lnu_into_fnu(
        lnu = lnu_peak.quantity,
        distance = dst_xrb.quantity
    )
    fnu_unit = cast(u.Unit,fnu_peak_q.unit)
    fnu_peak = QuantityData(
        value = np.asarray(fnu_peak_q.value,dtype=np.float64),
        unit = fnu_unit.to_string()
    )
    fnu_peak_value = fnu_peak.to_ndarray(conf["fnu_unit"])

    columns:list[str] = [
        "beta_sh",
        "a_wind",
        "phi_peak",
        "lnu_peak",
        "phi_theta",
        "lnu_theta",
        "tau_theta"
    ]

    df_output = pd.DataFrame(df_input[columns])
    df_output["fnu_net_peak"] = fnu_peak_value
    metadata_output = {
        **metadata_input,
        **conf
    }

    fw.write_csv_with_params(df_output,metadata_output,outpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "input",
        type=Path,
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
    args = parser.parse_args()
    main(args)
