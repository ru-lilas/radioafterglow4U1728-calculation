import numpy as np
from typing import Callable,cast
from module.types import FloatArray
import pandas as pd
from pathlib import Path
import warnings
from typing import Any
from ruamel.yaml import YAML

class Integrator:
    @staticmethod
    def trapezoid(
        x: FloatArray,
        y: FloatArray,
    ) -> float:
        return float(np.trapezoid(y, x))

class FileWriter:
    @staticmethod
    def df_to_csv(
        path: Path,
        df: pd.DataFrame,
        save_attrs:bool = True,
        save_index:bool = False
    ):
        metadata = df.attrs
        with open(path, "w") as f:
            if save_attrs:
                for key, value in metadata.items():
                    f.write(f"# {key}={value}\n")
            else:
                pass

            df.to_csv(f, index=save_index)

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

def ensure_yaml_mapping(
    yaml_raw:object
)->dict[str,Any]:
    if not isinstance(yaml_raw, dict):
        raise TypeError(
            "YAMLのトップレベルはmappingでなければなりません."
        )

    yaml_dict:dict[str,Any] = {}
    for key, value in yaml_raw.items():
        if not isinstance(key, str):
            raise TypeError(
                "YAMLのkeyは文字列(str)にしてください."
            )
        yaml_dict[key] = value

    return yaml_dict

class YAMLReader:
    @staticmethod
    def safe(
        path: Path
    )->dict[str,Any]:
        yaml_parser = YAML(
            typ="safe",
            pure=True
        )
        yaml_raw = yaml_parser.load(path)
        return ensure_yaml_mapping(yaml_raw)

class FileReader:

    @staticmethod
    def keyvalue(
        path: Path,
        split_sign:str="="
    ):
        keyvalue = {}
        if not path.is_file():
            print(f"ファイルが存在しません:{path}")
        #=== try to open csv file
        with open(path,"r", encoding="utf-8") as ofs:
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

    @classmethod
    def table_from_csv(
        cls,
        path: Path,
        idx:str|None = None,
        sep:str = ",",
        split_sign:str = "=",
    ):
        metadata = cls.keyvalue(path,split_sign=split_sign)
        df = pd.read_csv(
            path,
            index_col=None,
            comment = "#",
            sep = sep
        )
        if metadata is not None:
            df.attrs = metadata
        else:
            pass

        if idx is None:
            return df
        else:
            return df.set_index(idx)

    @staticmethod
    def yaml_safe(
        path: Path
    )->dict[str,Any]:
        return YAMLReader.safe(path)

class DataFrameUtils:

    @staticmethod
    def extract_row_by_index(
        df: pd.DataFrame,
        idx: int
    )->pd.Series:
        return cast(
            pd.Series,
            df.loc[idx]
        )
