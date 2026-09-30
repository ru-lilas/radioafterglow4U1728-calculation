"""
    inputs_as_dataclass.py
    インプットをdataclassで表現したものたち
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Self
import numpy as np

from module.types import FloatArray
from module import models,quantity_converter
from module.mydataclasses import (
    FloatVectorSpace,
    QuantityArray,
    QuantityArray1D,
    QuantityArray1DSpace,
    QuantityArrayBase,
    QuantityData,
    QuantityList,
    YAMLReadable,
    ValueArray
)

@dataclass(frozen=True,slots=True)
class DistanceData(QuantityData):
    distance_value: float
    distance_unit: str


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
class Chi2Configure:
    timewindow: SamplingTimewindow

@dataclass(frozen=True)
class CandidateModelsSelection:
    observation_frequencies: QuantityList

@dataclass(frozen=True,slots=True)
class ObservedBursterProperty:
    burster_name: str
    mdot_acc: QuantityData
    t_acc: QuantityData
    eps_nuc: QuantityData

    @property
    def m_acc(self)->QuantityData:
        mdot_acc = self.mdot_acc.quantity
        t_acc = self.t_acc.quantity
        m_acc = mdot_acc*t_acc
        return QuantityData.from_Quantity(m_acc)

    @property
    def e_nuc_max(self)->QuantityData:
        m_acc = self.m_acc.quantity
        eps_nuc = self.eps_nuc.quantity
        e_nuc_max = m_acc*eps_nuc
        return QuantityData.from_Quantity(e_nuc_max)

@dataclass(frozen=True,slots=True)
class Units:
    phi_unit: str
    fnu_unit: str

@dataclass(frozen=True,slots=True)
class ModelLightcurveConfig:
    xm: FloatVectorSpace
    time: QuantityArray1DSpace

@dataclass(frozen=True)
class GeneralInputs(YAMLReadable):
    units: Units
    model_parameters: models.InputParametersRaw
    candidate_models_selection: CandidateModelsSelection
    model_lightcurves: ModelLightcurveConfig
    chi2: Chi2Configure
    burster_property: ObservedBursterProperty

