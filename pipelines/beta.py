from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from module.dump_csv import dump_csv
from pipelines.build_input_params import config_builder
from pipelines.compute_spectrum import compute

base = OmegaConf.load("input/thermal_only/beta/base.yaml")
input_filepath = Path("input/thermal_only/beta/beta010.yaml")
beta = OmegaConf.load(input_filepath)

config = OmegaConf.merge(base, beta)

if not isinstance(config, DictConfig):
    raise TypeError(
        f"Expected DictConfig, got {type(config)}"
    )

input = config_builder(config)
input["beta_sh"] = config.beta_sh

metadata, df = compute(input)

dump_csv(metadata,df,config.output,input_filepath)
