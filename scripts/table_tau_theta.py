import argparse
from dataclasses import asdict
from pathlib import Path
from module import build_input_parameters
from module.utilities import filereaders as fr
from module.utilities import filewriters as fw
from tqdm import tqdm
import pandas as pd

def main(args:argparse.Namespace):
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    inpath:Path = args.input
    inputs:dict = fr.read_yaml(inpath)

    inputparams = build_input_parameters.both(inputs)

    outdata:list[dict] = []
    for inputparam in tqdm(inputparams):
        outdata.append({
            **asdict(inputparam),
            "tau_theta": inputparam.tau_theta,
            "phi_theta": inputparam.phi_theta.value,
            "phi_theta_unit": inputparam.phi_theta.unit,
            "l_theta": inputparam.l_theta.value,
            "l_unit": inputparam.l_theta.unit,
        })
    df = pd.DataFrame(outdata)
    fw.write_csv_with_params(df,{},outpath)
    print(f"output {outpath}")

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
