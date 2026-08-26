"""
    inputs_as_dataclass.py
    インプットをdataclassで表現したものたち
"""
from dataclasses import dataclass
from functools import cached_property
from module.mydataclasses import QuantityArrayBase, QuantityData,QuantityArray

@dataclass(frozen=True)
class SamplingTimewindow:
    min: float
    max: float
    unit: str

    @cached_property
    def t_min(self):
        return QuantityData(
            value=self.min,
            unit=self.unit
        )

    @cached_property
    def t_max(self):
        return QuantityData(
            value=self.max,
            unit=self.unit
        )

@dataclass(frozen=True)
class SamplingConfigure:
    nu: QuantityData
    timewindow: SamplingTimewindow
