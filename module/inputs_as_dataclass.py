"""
    inputs_as_dataclass.py
    インプットをdataclassで表現したものたち
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from module.types import FloatArray
from module import electron_temperature,synchrotron_scaling_values,quantity_converter
from module.mydataclasses import (
    QuantityArray,
    QuantityArrayBase,
    QuantityData,
    YAMLReadable,
    ValueArray
)

@dataclass(frozen=True,slots=True)
class PhysicalParametersBase:
    a_wind: QuantityArrayBase
    beta_sh: ValueArray
    eps_b: float
    eps_th: float
    mu: float
    mu_e: float
    distance: QuantityData


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

@dataclass(frozen=True,slots=True)
class PhysicalParameters:
    a_wind: QuantityArray
    beta_sh: FloatArray
    eps_b: float
    eps_th: float
    mu: float
    mu_e: float
    distance: QuantityData

    @classmethod
    def from_yaml(
        cls,
        path:Path
    )->Self:
        generalinput = GeneralInputs.from_yaml(path)
        indata = generalinput.physical_parameters
        return cls(
            a_wind = indata.a_wind.quantity_array,
            beta_sh = indata.beta_sh.arr,
            eps_b = indata.eps_b,
            eps_th = indata.eps_th,
            mu = indata.mu,
            mu_e = indata.mu_e,
            distance = indata.distance
        )

    @property
    def theta(self)->FloatArray:
        return electron_temperature.calculate_theta_e(
            eps_th=self.eps_th,
            mu=self.mu,
            mu_e=self.mu_e,
            beta_sh=self.beta_sh
        )

    @property
    def phi_theta(self):
        return synchrotron_scaling_values.calculate_phi_theta(
            theta = self.theta,
            eps_b = self.eps_b,
            a_wind = self.a_wind.quantity
        )

    @property
    def tau_theta(self)->FloatArray:
        return synchrotron_scaling_values.calculate_tau_theta(
            theta = self.theta,
            eps_b = self.eps_b,
            a_wind = self.a_wind.quantity,
            beta_sh=self.beta_sh,
            mu=self.mu,
            mu_e=self.mu_e,
        )

    @property
    def lnu_theta(self):
        return synchrotron_scaling_values.calculate_l_theta(
            theta = self.theta,
            eps_b = self.eps_b,
            a_wind = self.a_wind.quantity,
            beta_sh=self.beta_sh,
        )

    @property
    def doppler_delta(self):
        return quantity_converter.beta_into_doppler_delta(self.beta_sh)

@dataclass(frozen=True)
class GeneralInputs(YAMLReadable):
    physical_parameters: PhysicalParametersBase
    chi2fitting: Chi2FittingConfigure
