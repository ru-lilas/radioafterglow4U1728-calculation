import pandas as pd
from module.utilities import filewriters as fw
from omegaconf import DictConfig
from pathlib import Path

def build_output_path(config_output:DictConfig,input_filepath:Path)->Path:
    output_dir = Path(config_output.dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    input_filename = input_filepath.stem
    return Path(output_dir/f"{input_filename}.csv")

def dump_csv(
    metadata:dict,
    df:pd.DataFrame,
    config_output:DictConfig,
    input_filepath:Path
)->Path:
    output_path = build_output_path(config_output,input_filepath)
    fw.write_csv_with_params(df,metadata,output_path)
    print(f"output {output_path}")
    return output_path

