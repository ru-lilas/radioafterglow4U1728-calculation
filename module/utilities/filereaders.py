from typing import Any
import yaml
import pandas as pd
from pathlib import Path
import warnings

def read_csv(filepath:Path,sep:str=","):
    df = pd.read_csv(
        filepath,
        index_col=None,
        comment = "#",
        sep=sep,
    )
    return df

def read_yaml(filepath: Path):
    with open(filepath,"r") as f:
        config = yaml.safe_load(f)
    return config

def parse_value(value: str) -> Any:
    value = value.strip()

    # 空文字だけは warning
    if value == "":
        warnings.warn("empty value detected")
        return value

    # floatとして読めるならfloat
    try:
        return float(value)

    # 失敗したら普通にstr
    except ValueError:
        return value

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

                key = key.strip()
                keyvalue[key] = parse_value(value)
        else:
            break
    return keyvalue
