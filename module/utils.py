import numpy as np
from typing import Callable
from module.types import FloatArray
import pandas as pd
from pathlib import Path

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
