import yaml
import pandas as pd
from pathlib import Path

def write_dict_as_yaml(params:dict,filepath: Path):
    with open(filepath, "w") as f:
        yaml.dump(params, f)

def write_csv_with_params(
    df: pd.DataFrame,
    fixed_params: dict,
    filepath: Path,
    use_index:bool = False
) -> None:

    with open(filepath, "w") as f:
        for key, value in fixed_params.items():
            f.write(f"# {key}={value}\n")

        if use_index:
            df.to_csv(f, index=use_index,index_label="idx")
        else:
            df.to_csv(f, index=use_index)

def write_csv(
    df: pd.DataFrame,
    filepath: Path,
    use_index:bool = False
) -> None:

    with open(filepath, "w") as f:
        if use_index:
            df.to_csv(f, index=use_index,index_label="idx")
        else:
            df.to_csv(f, index=use_index)
