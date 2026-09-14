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
    QuantityArrayBase,
    QuantityData,
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
class Chi2FittingConfigure:
    free_parameters: list
    sampling: SamplingConfigure
    model: ModelConfigure

    @property
    def n_model(self):
        return len(self.free_parameters)


# @dataclass(frozen=True,slots=True)
# class ModelParameterGrid:
#     a_wind: QuantityArray
#     beta_sh: FloatArray
#     eps_b: float
#     eps_th: float
#     mu: float
#     mu_e: float
#     distance: QuantityData
#
#     @classmethod
#     def from_yaml(
#         cls,
#         path:Path
#     )->Self:
#         generalinput = GeneralInputs.from_yaml(path)
#         indata = generalinput.model_parameters
#         return cls(
#             a_wind = indata.a_wind.quantity_array,
#             beta_sh = indata.beta_sh.arr,
#             eps_b = indata.eps_b,
#             eps_th = indata.eps_th,
#             mu = indata.mu,
#             mu_e = indata.mu_e,
#             distance = indata.distance
#         )
#
#     @property
#     def a_wind_axis(self)->FloatArray:
#         return np.unique(self.a_wind.values)
#
#     @property
#     def beta_sh_axis(self)->FloatArray:
#         return np.unique(self.beta_sh)
#
#     def build_meshgrid(self):
#         a_wind_unit = self.a_wind.unit
#
#         a_wind_value_grid,beta_sh_grid = np.meshgrid(
#             self.a_wind_axis,
#             self.beta_sh_axis,
#             indexing="xy",
#             sparse=True
#         )
#         a_wind_grid = QuantityArray(
#             values = a_wind_value_grid,
#             unit = a_wind_unit
#         )
#         return (a_wind_grid,beta_sh_grid)
#
#     @property
#     def theta(self)->FloatArray:
#         return electron_temperature.calculate_theta_e(
#             eps_th=self.eps_th,
#             mu=self.mu,
#             mu_e=self.mu_e,
#             beta_sh=self.beta_sh
#         )
#
#     @property
#     def phi_theta(self):
#         return synchrotron_scaling_values.calculate_phi_theta(
#             theta = self.theta,
#             eps_b = self.eps_b,
#             a_wind = self.a_wind.quantity
#         )
#
#     @property
#     def tau_theta(self)->FloatArray:
#         return synchrotron_scaling_values.calculate_tau_theta(
#             theta = self.theta,
#             eps_b = self.eps_b,
#             a_wind = self.a_wind.quantity,
#             beta_sh=self.beta_sh,
#             mu=self.mu,
#             mu_e=self.mu_e,
#         )
#
#     @property
#     def lnu_theta(self):
#         return synchrotron_scaling_values.calculate_l_theta(
#             theta = self.theta,
#             eps_b = self.eps_b,
#             a_wind = self.a_wind.quantity,
#             beta_sh=self.beta_sh,
#         )
#
#     @property
#     def doppler_delta(self):
#         return quantity_converter.beta_into_doppler_delta(self.beta_sh)


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
    mu: float
    mu_e: float
    distance: QuantityData

    def build(
        self
    )->ModelParameterTable:
        return ModelParameterTable(
            grid = ParameterGrid.from_arrays(
                phi_th=self.phi_th.to_QuantityArray1D(),
                theta0=self.theta0.to_floatarray()
            ),
            eps_b = self.eps_b,
            eps_th = self.eps_th,
            mep = models.MolecularWeightRatio.from_params(
                mu=self.mu,
                mu_e=self.mu_e
            ).to_mep(),
            distance= self.distance
        )

@dataclass(frozen=True)
class GeneralInputs(YAMLReadable):
    model_parameters: ModelParameters
    chi2fitting: Chi2FittingConfigure
    burster_property: ObservedBursterProperty

