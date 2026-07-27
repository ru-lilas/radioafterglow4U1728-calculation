from argparse import Namespace
import argparse
from pathlib import Path
from numpy.typing import NDArray
from module import synchrotron_function
from module.utilities import filereaders as fr
from module.mydataclasses import ValueArray
import numpy as np
import pandas as pd

def main(args:Namespace):
    inpath:Path = args.input
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    conf = ValueArray.from_yaml(inpath)
    x_arr = conf.arr

    ipxi = synchrotron_function.thermal_Ip(x_arr)

    df = pd.DataFrame({
        "xm":x_arr,
        "ln_ip":np.log(ipxi),
    })

    df.to_csv(outpath,index=False)
    print(f"output {outpath}")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "input",
        type=Path
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    main(args)

