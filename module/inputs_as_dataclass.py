"""
    inputs_as_dataclass.py
    インプットをdataclassで表現したものたち
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from module.mydataclasses import QuantityArrayBase, QuantityData,QuantityArray

@dataclass(frozen=True)
class SamplingTimewindow:
    min: float
    max: float
    unit: str

    @property
    def t_min(self):
        return QuantityData(
            value=self.min,
            unit=self.unit
        )

    @property
    def t_max(self):
        return QuantityData(
            value=self.max,
            unit=self.unit
        )

@dataclass(frozen=True,slots=True)
class SamplingConfigure:
    nu: QuantityData
    timewindow: SamplingTimewindow

@dataclass(frozen=True,slots=True)
class ModelConfigure:
    time: QuantityArrayBase
    fnu_unit: str
    nu: QuantityData

    @property
    def t_obs(self):
        t = QuantityArray(
            values = self.time.values.arr,
            unit=self.time.unit
        )
        return t.quantity
    
    @property
    def nu_obs(self):
        return self.nu.quantity

@dataclass(frozen=True)
class Chi2FittingConfigure:
    free_parameters: list
    model: ModelConfigure
    sampling: SamplingConfigure

    @property
    def n_model(self):
        return len(self.free_parameters)
