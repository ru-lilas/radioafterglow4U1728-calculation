from dataclasses import dataclass

from dacite import from_dict
from pathlib import Path
import pandas as pd

@dataclass
class LightcurveData:
    nu: float
    df: pd.DataFrame

