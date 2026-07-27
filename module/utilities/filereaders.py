from typing import Any, cast
import yaml
import pandas as pd
from pathlib import Path
import warnings
from ruamel.yaml import YAML, CommentedMap
from typing import Self
from dacite import from_dict
from typing import TypeVar

def read_csv(filepath:Path,sep:str=","):
    df = pd.read_csv(
        filepath,
        index_col=None,
        comment = "#",
        sep=sep,
    )
    return df

def read_csv_within_idx(filepath:Path,idx_name:str="idx",sep:str=","):
    df = pd.read_csv(
        filepath,
        index_col=None,
        comment = "#",
        sep=sep,
    )
    return df.set_index(idx_name)

def read_yaml_pyyaml(filepath: Path):
    with open(filepath,"r") as f:
        config = yaml.safe_load(f)
    return config

def read_yaml(path: Path):
    yamlperser = YAML()
    with open(path,"r",encoding='utf-8') as f:
        config = cast(CommentedMap,yamlperser.load(f))
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

T = TypeVar("T")
class InputReader:
    @staticmethod
    def read(path: Path, data_class: type[T]) -> T:
        dict_data = read_yaml_pyyaml(path)
        return from_dict(
            data_class=data_class,
            data=dict_data,
        )

class YAMLReadable:
    @classmethod
    def from_yaml(
            cls,
            path:Path
    )-> Self:
        return InputReader.read(
            path,cls
        )

