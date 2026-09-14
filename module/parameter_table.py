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
    dataframe_utils,
    electron_temperature,
    models,
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
    PHI_TH = auto()
    THETA0 = auto()
    EPS_B = auto()
    EPS_TH = auto()
    MU_E_MU = auto()
    DISTANCE_VALUE = auto()
    DISTANCE_UNIT = auto()
    A_WIND_UNIT = auto()
    PHI_UNIT = auto()
    LNU_UNIT = auto()
    FNU_UNIT = auto()

class ModelColumnNames(StrEnum):
    PHI_PEAK = auto()
    FNU_PEAK = auto()
    THETA = auto()
    THETA_THIN = auto()
    IS_SSA_THICK = auto()
    A_SYN = auto()
    PHI_A = auto()
    PHI_THETA = auto()
    TAU_THETA = auto()
    LNU_THETA = auto()
    DOPPLER_DELTA = auto()

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

@dataclass(frozen=True,slots=True)
class ParameterGrid:
    phi_th:models.ThermalSynchrotronPhase
    theta0:models.ElectronTemperatureScaling

    @classmethod
    def from_arrays(
        cls,
        phi_th:QuantityArray,
        theta0:FloatArray
    )->Self:
        phith_val = phi_th.values
        phi_unit = phi_th.unit
        phith_axis = np.unique(phith_val)

        theta0_axis = np.unique(theta0)
        phith_value_grid,theta0_grid = np.meshgrid(
            phith_axis,
            theta0_axis,
            indexing="xy",
            sparse=False
        )
        phith_grid = models.ThermalSynchrotronPhase(
            values=phith_value_grid,
            unit=phi_unit
        )
        return cls(
            phi_th=phith_grid,
            theta0=models.ElectronTemperatureScaling(theta0_grid)
        )

@dataclass(frozen=True,slots=True)
class ModelParameterTable:
    grid: ParameterGrid
    eps_b: float
    eps_th: float
    mep: float
    distance: QuantityData

    @property
    def beta_sh(self):
        return self.grid.theta0.to_beta(
            eps_th=self.eps_th,
            mep=self.mep
        )

    @property
    def theta(self):
        return self.grid.theta0.to_theta()

    @property
    def theta_thin(self):
        return models.ElectronTemperatureOpticallyThinLimit.from_params(
            mep=self.mep,
            eps_b=self.eps_b,
            phi_th=self.grid.phi_th
        )

    @property
    def tau_theta(self)->models.OpticalDepthParameter:
        return models.OpticalDepthParameter.from_params_exact(
            mep=self.mep,
            eps_b=self.eps_b,
            theta=self.theta,
            phi_th=self.grid.phi_th
        )

    @property
    def lnu_theta(self):
        return models.ThermalSynchrotronLuminosity.from_params(
            mep=self.mep,
            eps_b=self.eps_b,
            theta=self.theta,
            eps_th=self.eps_th,
            phi_th=self.grid.phi_th
        )

    @property
    def doppler_delta(self):
        return quantity_converter.beta_into_doppler_delta(self.beta_sh)

    def phi_peak(
        self,
        peak_table:LambdaPeakTable
    )->QuantityArray:
        phi_theta = self.grid.phi_th
        xm_peak = peak_table.interpolated_lambda_peak(
            self.tau_theta.values
        )
        return QuantityArray(
            values = phi_theta.values*xm_peak,
            unit = phi_theta.unit
        )

    def lnu_peak(
        self,
        peak_table:LambdaPeakTable
    )->QuantityArray:
        lnu_theta = self.lnu_theta
        lambda_peak = peak_table.interpolated_lambda_peak(
            self.tau_theta.values
        )
        return QuantityArray(
            values = lnu_theta.values*lambda_peak,
            unit = lnu_theta.unit
        )

    def fnu_peak_observerframe(
        self,
        peak_table:LambdaPeakTable
    )->QuantityArray:
        lnu_peak = self.lnu_peak(peak_table).quantity
        distance = self.distance.quantity
        doppler_delta = self.doppler_delta 
        fnu_peak = doppler_delta**3 * lnu_peak / (4.0*np.pi*distance**2)
        return QuantityArray.from_Quantity(
            quantity = fnu_peak
        )

    def to_df(
        self,
        peak_table:LambdaPeakTable,
        *,
        a_wind_unit:str = "g cm-1",
        phi_unit:str = "GHz min",
        lnu_unit:str = "erg s-1 Hz-1",
        fnu_unit:str = "mJy",
    )->pd.DataFrame:
        grid = self.grid
        theta_val = self.theta.values
        theta_thin_val = self.theta_thin.values
        phi_peak = self.phi_peak(peak_table)
        fnu_peak = self.fnu_peak_observerframe(peak_table)
        df = pd.DataFrame({
            ModelParameterNames.PHI_TH: grid.phi_th.to_value_in(phi_unit).ravel(),
            ModelParameterNames.THETA0: grid.theta0.values.ravel(),
            ModelColumnNames.THETA: theta_val.ravel(),
            ModelColumnNames.THETA_THIN: theta_thin_val.ravel(),
            ModelColumnNames.IS_SSA_THICK: (theta_val < theta_thin_val).ravel(),
            ModelColumnNames.TAU_THETA: self.tau_theta.values.ravel(),
            ModelColumnNames.LNU_THETA: self.lnu_theta.to_value_in(lnu_unit).ravel(),
            ModelColumnNames.DOPPLER_DELTA: self.doppler_delta.ravel(),
            ModelColumnNames.PHI_PEAK: phi_peak.to_value_in(phi_unit).ravel(),
            ModelColumnNames.FNU_PEAK: fnu_peak.to_value_in(fnu_unit).ravel()
        })
        df.attrs[ModelParameterNames.EPS_B] = self.eps_b
        df.attrs[ModelParameterNames.EPS_TH] = self.eps_th
        df.attrs[ModelParameterNames.MU_E_MU] = models.MolecularWeightRatio.from_mep(self.mep)
        df.attrs[ModelParameterNames.DISTANCE_VALUE] = self.distance.value
        df.attrs[ModelParameterNames.DISTANCE_UNIT] = self.distance.unit
        df.attrs[ModelParameterNames.A_WIND_UNIT] = a_wind_unit
        df.attrs[ModelParameterNames.PHI_UNIT] = phi_unit
        df.attrs[ModelParameterNames.LNU_UNIT] = lnu_unit
        df.attrs[ModelParameterNames.FNU_UNIT] = fnu_unit
        return df

