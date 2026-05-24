from pathlib import Path
from typing import Any
from module.dump_csv import dump_csv
from module.utilities import filereaders as fr
from module.models import InputParameters, SynchrotronSpectrum, Frequency
from module import build_output_data
import argparse
import numpy as np
from tqdm import tqdm

def _build_time_array(input_raw:dict)->dict[str,Any]:
    try:
        input_time = input_raw["time"]
        t_unit = input_time["unit"]
    except KeyError as e:
        available = ", ".join(input_raw.keys())

        raise KeyError(
            f"{e.args[0]} not found. "
            f"Available keys: {available}"
        ) from e
    return {
        "t_array_value": np.logspace(
            start=input_time["log10_min"],
            stop=input_time["log10_max"],
            num=input_time["num"]
        ),
        "t_unit": t_unit
    }

def _build_freq_array(input_raw:dict)->dict[str,Any]:
    try:
        input_freq = input_raw["frequency"]
    except KeyError as e:
        available = ", ".join(input_raw.keys())

        raise KeyError(
            f"{e.args[0]} not found. "
            f"Available keys: {available}"
        ) from e
    return {
        "value_array": np.logspace(
            start=input_freq["log10_min"],
            stop=input_freq["log10_max"],
            num=input_freq["num"]
        ),
        "unit": input_freq["unit"]
    }

def main(args:argparse.Namespace):

    outdir:Path = args.outdir
    outdir.mkdir(parents=True,exist_ok=True)

    input_raw = fr.read_yaml(args.input)
    input_time = _build_time_array(input_raw)
    input_freq = _build_freq_array(input_raw)

    t_array = np.asarray(input_time["t_array_value"])
    input_params = InputParameters(
        **{
            **input_raw["plasma"],
            **input_raw["wind"],
            "beta_sh": input_raw["beta_sh"],
        }
    )
    nu = Frequency(**input_freq)
    ss = SynchrotronSpectrum(
        inputparams=input_params,
        t_value = float(input_time["t_array_value"][0]),
        t_unit = input_time["t_unit"],
        nu=nu
        )

    for i,t in enumerate(tqdm(t_array)):
        ss.t_value = t
        metadata = build_output_data.metadata(
            inputparams=input_params,
            ss=ss
        )
        df = build_output_data.tabledata(ss)
        dump_csv(metadata,df,outdir/f"{i:03d}.csv")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "input",
        type=Path,
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True
    )
    
    args = parser.parse_args()
    
    main(args)
