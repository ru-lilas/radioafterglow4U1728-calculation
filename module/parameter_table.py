"""
    parameter_table.py
    inputからmodelごとのパラメータ表を作成する
"""

from pathlib import Path
from typing import Self, cast
from dataclasses import dataclass
from module.dataframe_processors import convert_ndarray
from module.types import FloatArray
from module import (
    electron_temperature,
    synchrotron_scaling_values,
    quantity_converter)
import numpy as np
import pandas as pd
from module import mydataclasses, synchrotron_scaling_values
from enum import StrEnum, auto
from module.mydataclasses import (
    QuantityArray,
    QuantityArray1D,
    QuantityArrayBase,
    QuantityData,
    YAMLReadable,
    ValueArray
)
from module.utils import FileReader, FileWriter

class SynchrotronPeakColumns(StrEnum):
    TAU_THETA = auto()
    XM_PEAK = auto()
    LAMBDA_PEAK = auto()

class ModelParameterNames(StrEnum):
    A_WIND = auto()
    BETA_SH = auto()
    EPS_B = auto()
    EPS_TH = auto()
    MU = auto()
    MU_E = auto()
    DISTANCE_VALUE = auto()
    DISTANCE_UNIT = auto()
    THETA = auto()
    PHI_THETA = auto()
    TAU_THETA = auto()
    LNU_THETA = auto()
    DOPPLER_DELTA = auto()
    A_WIND_UNIT = auto()
    PHI_UNIT = auto()
    LNU_UNIT = auto()

@dataclass(frozen=True,slots=True)
class ModelParameterTable:
    a_wind: QuantityArray1D
    beta_sh: FloatArray
    eps_b: float
    eps_th: float
    mu: float
    mu_e: float
    distance: QuantityData

    @property
    def a_wind_axis(self)->FloatArray:
        return np.unique(self.a_wind.values)

    @property
    def beta_sh_axis(self)->FloatArray:
        return np.unique(self.beta_sh)

    def build_meshgrid(self):
        a_wind_unit = self.a_wind.unit

        a_wind_value_grid,beta_sh_grid = np.meshgrid(
            self.a_wind_axis,
            self.beta_sh_axis,
            indexing="xy",
            sparse=True
        )
        a_wind_grid = QuantityArray(
            values = a_wind_value_grid,
            unit = a_wind_unit
        )
        return (a_wind_grid,beta_sh_grid)

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
        return QuantityArray1D.from_Quantity(
            synchrotron_scaling_values.calculate_phi_theta(
                theta = self.theta,
                eps_b = self.eps_b,
                a_wind = self.a_wind.quantity
        ))

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
        return QuantityArray1D.from_Quantity(
            synchrotron_scaling_values.calculate_l_theta(
                theta = self.theta,
                eps_b = self.eps_b,
                a_wind = self.a_wind.quantity,
                beta_sh=self.beta_sh,
        ))

    @property
    def doppler_delta(self):
        return quantity_converter.beta_into_doppler_delta(self.beta_sh)

    @classmethod
    def from_csv(
        cls,
        path:Path
    ):
        df = FileReader.table_from_csv(path,idx="index")
        metadata = df.attrs
        return

    def to_df(
        self,
        *,
        a_wind_unit:str = "g cm-1",
        phi_unit:str = "GHz min",
        lnu_unit:str = "erg s-1 Hz-1"
    )->pd.DataFrame:
        a_wind = self.a_wind
        phi_theta = self.phi_theta
        lnu_theta = self.lnu_theta
        df = pd.DataFrame({
            ModelParameterNames.A_WIND: a_wind.to_value_in(a_wind_unit),
            ModelParameterNames.BETA_SH: self.beta_sh,
            ModelParameterNames.THETA: self.theta,
            ModelParameterNames.PHI_THETA: phi_theta.to_value_in(phi_unit),
            ModelParameterNames.LNU_THETA: lnu_theta.to_value_in(lnu_unit),
            ModelParameterNames.TAU_THETA: self.tau_theta,
            ModelParameterNames.DOPPLER_DELTA: self.doppler_delta
        })
        df.attrs[ModelParameterNames.EPS_B] = self.eps_b
        df.attrs[ModelParameterNames.EPS_TH] = self.eps_th
        df.attrs[ModelParameterNames.MU] = self.mu
        df.attrs[ModelParameterNames.MU_E] = self.mu_e
        df.attrs[ModelParameterNames.DISTANCE_VALUE] = self.distance.value
        df.attrs[ModelParameterNames.DISTANCE_UNIT] = self.distance.unit
        df.attrs[ModelParameterNames.A_WIND_UNIT] = a_wind_unit
        df.attrs[ModelParameterNames.PHI_UNIT] = phi_unit
        df.attrs[ModelParameterNames.LNU_UNIT] = lnu_unit
        return df

@dataclass(frozen=True,slots=True)
class LambdaPeakTable:
    tau_theta_ref: FloatArray
    xm_peak_ref: FloatArray
    lambda_peak_ref: FloatArray

    @classmethod
    def from_csv(
            cls,
            path:Path
    )->Self:
        df = FileReader.table_from_csv(path)
        return cls(
            tau_theta_ref = convert_ndarray(df,SynchrotronPeakColumns.TAU_THETA),
            xm_peak_ref = convert_ndarray(df,SynchrotronPeakColumns.XM_PEAK),
            lambda_peak_ref = convert_ndarray(df,SynchrotronPeakColumns.LAMBDA_PEAK),
        )

    def interpolated_xm_peak(
        self,
        tau_theta:FloatArray
    )->FloatArray:
        return np.interp(
            tau_theta,
            self.tau_theta_ref,
            self.xm_peak_ref
        )

    def interpolated_lambda_peak(
        self,
        tau_theta:FloatArray
    )->FloatArray:
        return np.interp(
            tau_theta,
            self.tau_theta_ref,
            self.lambda_peak_ref
        )
