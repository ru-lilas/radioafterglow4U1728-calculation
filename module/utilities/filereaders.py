from typing import Any
import yaml
import pandas as pd
from pathlib import Path

def read_csv(filepath:Path):
    df = pd.read_csv(
            filepath,
            index_col=None,
            comment = "#",
    )
    return df

def read_yaml(filepath: Path):
    with open(filepath,"r") as f:
        config = yaml.safe_load(f)
    return config

def read_keyvalue(filepath:Path,split_sign:str="=")->dict[str,Any]:
    keyvalue = {}
    if not filepath.is_file():
        print(f"ファイルが存在しません:{filepath}")
    #=== try to open csv file
    with open(filepath,"r", encoding="utf-8") as ofs:
        lines = ofs.readlines()
    for line in lines:
        line = line.strip()  # remove whitespace
        if line.startswith('#'):
            line = line[1:].strip()
            if split_sign in line:
                key, value = line.split(split_sign,1)
                try:
                    keyvalue[key.strip()] = float(value.strip())
                except ValueError:
                    print(f"Warning: fail to convert value into float type:metadata['{key}'] = {value}")
                    keyvalue[key] = value.strip()
        else:
            break
    return keyvalue
