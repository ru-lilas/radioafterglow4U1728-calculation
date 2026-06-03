from functools import cached_property
from pathlib import Path

from numpy.typing import NDArray
from module.utilities import filereaders as fr
import numpy as np
import pandas as pd
from dataclasses import dataclass

def read_tabular(input:Path):
    tabular = fr.read_csv(input)
    return tabular

@dataclass
class ThermalSynchrotronTable:
    tabular: pd.DataFrame

    @cached_property
    def xi_tabular(self):
        return np.array(self.tabular["xi"], dtype=np.float64)

    @cached_property
    def log_ip_tabular(self):
        return np.array(self.tabular["ln_ip"], dtype=np.float64)

    def calculate_log_ip(self,xi:NDArray[np.float64])->NDArray[np.float64]:
        return np.interp(x=xi,xp=self.xi_tabular,fp=self.log_ip_tabular)
