"""
    inputs_as_dataclass.py
    インプットをdataclassで表現したものたち
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Self
import numpy as np

from module.parameter_table import ModelParameterTable, ParameterGrid
from module.types import FloatArray
from module import electron_temperature, models,synchrotron_scaling_values,quantity_converter
from module.mydataclasses import (
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
class ModelParameters:
    phi_th: QuantityArrayBase
    theta0: ValueArray
    eps_b: float
    eps_th: float
    molecular_weight: models.MolecularWeight
    distance: QuantityData
    powerlaw: models.PowerlawParameters

    @property
    def phi_th_array(self):
        return self.phi_th.to_QuantityArray1D()

    @property
    def theta0_array(self):
        return self.theta0.to_floatarray()

    def build_grid(self):
        return ParameterGrid.from_arrays(
            self.phi_th_array,
            self.theta0_array
        )

    def to_df(
        self
    )->ModelParameterTable:
        return ModelParameterTable(
            grid = ParameterGrid.from_arrays(
                phi_th=self.phi_th_array,
                theta0=self.theta0.to_floatarray()
            ),
            eps_b = self.eps_b,
            eps_th = self.eps_th,
            mep = self.molecular_weight.to_mass_ratio(),
            distance= self.distance
        )
@dataclass(frozen=True,slots=True)
class Units:
    phi_unit: str
    fnu_unit: str

@dataclass(frozen=True,slots=True)
class ModelLightcurveConfig:
    time: QuantityArray1DSpace

@dataclass(frozen=True)
class GeneralInputs(YAMLReadable):
    units: Units
    model_parameters: ModelParameters
    candidate_models_selection: CandidateModelsSelection
    model_lightcurves: ModelLightcurveConfig
    chi2: Chi2Configure
    burster_property: ObservedBursterProperty

