import yaml
import pandas as pd
from pathlib import Path

def write_dict_as_yaml(params:dict,filepath: Path):
    with open(filepath, "w") as f:
        yaml.dump(params, f)

def write_csv_with_params(
    df: pd.DataFrame,
    fixed_params: dict,
    filepath: Path
) -> None:

    with open(filepath, "w") as f:
        for key, value in fixed_params.items():
            f.write(f"# {key}={value}\n")

        df.to_csv(f, index=False)
