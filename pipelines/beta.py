from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from module.dump_csv import dump_csv
from pipelines.build_input_params import config_builder
from pipelines.compute_spectrum import compute
import argparse

def main(args:argparse.Namespace):

    inputpath = args.infile
    base = OmegaConf.load("input/thermal_only/beta/base.yaml")
    beta = OmegaConf.load(inputpath)
    
    config = OmegaConf.merge(base, beta)
    
    if not isinstance(config, DictConfig):
        raise TypeError(
            f"Expected DictConfig, got {type(config)}"
        )
    
    input = config_builder(config)
    input["beta_sh"] = config.beta_sh
    
    metadata, df = compute(input)
    
    dump_csv(metadata,df,args.outpath)
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "infile",
        type=Path
    )
    parser.add_argument(
        "outpath",
        type=Path
    )
    
    args = parser.parse_args()
    
    print(f"input {args.infile}")
    main(args)
