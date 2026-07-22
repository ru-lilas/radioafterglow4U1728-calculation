from dataclasses import dataclass

from module.utilities import filereaders as fr
from dacite import from_dict
from pathlib import Path
from module import input_dataclasses
from typing import TypeVar, Literal
import numpy as np
import pandas as pd

type ValueScale = Literal["linear","log"]

def read_physical_parameters(path:Path):
    dict_data:dict = fr.read_yaml(path)
    return from_dict(
        data_class=input_dataclasses.PhysicalParameters,
        data=dict_data
    )

def read_sampling(path:Path):
    dict_data:dict = fr.read_yaml(path)
    return from_dict(
        data_class=input_dataclasses.SAMPLING,
        data=dict_data
    )

def read_freeparameters(path:Path):
    dict_data:dict = fr.read_yaml(path)
    return from_dict(
        data_class=input_dataclasses.FreeParameters,
        data=dict_data
    )
@dataclass
class Micprophysics:
    eps_th: float
    eps_b: float
    mu: float
    mu_e: float

@dataclass
class ValueArray:
    start: float
    stop: float
    num: int
    scale: str

    @property
    def nparr(self):
        if (self.scale == "linear"):
            return np.linspace(start=self.start,stop=self.stop,num=self.num,dtype=np.float64)
        else:
            return np.logspace(start=self.start,stop=self.stop,num=self.num,dtype=np.float64)

@dataclass
class LightcurveInput:
    a_wind_value: float 
    a_wind_unit: str
    beta_sh: float
    nu: float
    nu_unit: str
    microphys: Micprophysics

@dataclass
class LightcurveConfigure:
    t_value_arr: ValueArray
    t_unit: str
    fnu_unit: str

T = TypeVar("T")

class InputReader:
    @staticmethod
    def read(path: Path, data_class: type[T]) -> T:
        dict_data = fr.read_yaml(path)
        return from_dict(
            data_class=data_class,
            data=dict_data,
        )
