from pathlib import Path
from dataclasses import dataclass
from functools import cached_property
from typing import cast
from module.mydataclasses import QuantityData,QuantityArray
from module.types import FloatArray
from module.utilities import filereaders as fr
import astropy.units as u
import numpy as np
import pandas as pd
from module import mydataclasses, synchrotron_scaling_values
from module import electron_temperature
from enum import StrEnum, auto
from module import quantity_converter

class Columns(StrEnum):
    A_WIND = auto()
    BETA_SH = auto()
    EPS_B = auto()
    EPS_TH = auto()
    MU = auto()
    MU_E = auto()
    DISTANCE = auto()
    THETA = auto()
    PHI_THETA = auto()
    TAU_THETA = auto()
    LNU_THETA = auto()
    DOPPLER_DELTA = auto()

@dataclass(frozen=True,slots=True)
class PhysicalParameters:
    a_wind: mydataclasses.QuantityArray
    beta_sh: mydataclasses.ValueArray
    eps_b: float
    eps_th: float
    mu: float
    mu_e: float
    distance: mydataclasses.QuantityData

    @property
    def theta(self)->FloatArray:
        return electron_temperature.calculate_theta_e(
            eps_th=self.eps_th,
            mu=self.mu,
            mu_e=self.mu_e,
            beta_sh=self.beta_sh.arr
        )

    @property
    def phi_theta(self)->u.Quantity:
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
            beta_sh=self.beta_sh.arr,
            mu=self.mu,
            mu_e=self.mu_e,
        )

    @property
    def lnu_theta(self)->u.Quantity:
        return synchrotron_scaling_values.calculate_l_theta(
            theta = self.theta,
            eps_b = self.eps_b,
            a_wind = self.a_wind.quantity,
            beta_sh=self.beta_sh.arr,
        )

    @property
    def doppler_delta(self):
        return quantity_converter.beta_into_doppler_delta(self.beta_sh.arr)

    def to_df(self):
        return pd.DataFrame({
            Columns.A_WIND: self.a_wind.values.arr,
            Columns.BETA_SH: self.beta_sh.arr,
            Columns.EPS_B: self.eps_b,
            Columns.EPS_TH: self.eps_th,
            Columns.MU: self.mu,
            Columns.MU_E: self.mu_e,
            Columns.DISTANCE: self.distance.value,
            Columns.THETA: self.theta,
            Columns.PHI_THETA: self.phi_theta.value,
            Columns.LNU_THETA: self.lnu_theta.value,
            Columns.DOPPLER_DELTA: self.doppler_delta
        })

    def metadata(self):
        return {
            Columns.A_WIND: self.a_wind.unit,
            Columns.DISTANCE: self.distance.unit,
            Columns.PHI_THETA: cast(u.UnitBase,self.phi_theta.unit).to_string(),
            Columns.LNU_THETA: cast(u.UnitBase,self.lnu_theta.unit).to_string()
        }
