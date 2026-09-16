"""
    parameter_table.py
    inputからmodelごとのパラメータ表を作成する
"""

from os import wait
from pathlib import Path
from typing import Self, cast
from dataclasses import dataclass

from numpy.typing import NDArray
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
from module.synchrotron_numerical_table import DimensionlessThermalPeak
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
    PHI_TH_THIN = auto()
    PHI_PEAK = auto()
    FNU_PEAK = auto()
    THETA = auto()
    THETA_THIN = auto()
    IS_SSA_THICK = auto()
    A_SYN = auto()
    A_WIND = auto()
    BETA_SH = auto()
    PHI_A = auto()
    PHI_THETA = auto()
    TAU_THETA = auto()
    LNU_THETA = auto()
    DOPPLER_DELTA = auto()
    TAU_TH_THIN = auto()
    LNU_TH_THIN = auto()
    PHI_PEAK_THIN = auto()
    FNU_PEAK_THIN = auto()

class ModelOpticallythinColumnNames(StrEnum):
    PHI_TH_THIN = auto()
    TAU_TH_THIN = auto()
    LNU_TH_THIN = auto()
    PHI_PEAK_THIN = auto()
    FNU_PEAK_THIN = auto()

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

    def beta_sh(self):
        return self.grid.theta0.to_beta(
            eps_th=self.eps_th,
            mep=self.mep
        )

    def theta(self):
        return self.grid.theta0.to_theta()

    def a_syn(
            self,
            theta:models.ElectronTemperature
        )->QuantityArray:
        theta_val = theta.values
        return models.WindParameterNormalization.synchrotron(
            theta=theta_val
        )

    def a_wind(
        self,
        a_syn:QuantityArray
    )->models.WindParameter:
        return models.WindParameter.from_phi_th(
            eps_b=self.eps_b,
            a_syn=a_syn,
            phi_th=self.grid.phi_th
        )

    @property
    def theta_thin(self):
        theta0 = models.SelfAbsorptionOpticallyThinLimit.theta0(
            eps_b=self.eps_b,
            eps_th=self.eps_th,
            phith=self.grid.phi_th,
            mep=self.mep
        )
        return theta0

    def tau_theta(
        self,
        theta:models.ElectronTemperature
    )->models.OpticalDepthParameter:
        return models.OpticalDepthParameter.from_params_exact(
            mep=self.mep,
            eps_b=self.eps_b,
            eps_th=self.eps_th,
            theta=theta,
            theta0=self.grid.theta0,
            phi_th=self.grid.phi_th
        )

    def phi_th_thin(
        self,
        theta:models.ElectronTemperature
    )->models.ThermalSynchrotronPhase:
        return models.ThermalSynchrotronPhase.optically_thin_limit(
            eps_b=self.eps_b,
            eps_th=self.eps_th,
            mep=self.mep,
            theta=theta,
            theta0=self.grid.theta0
        )

    def tau_theta_thinlimit(
        self,
        theta:models.ElectronTemperature,
        phi_th_thin: models.ThermalSynchrotronPhase
    )->models.OpticalDepthParameter:
        """
            order unity
        """
        return models.OpticalDepthParameter.from_params_exact(
            mep=self.mep,
            eps_b=self.eps_b,
            eps_th=self.eps_th,
            theta=theta,
            theta0=self.grid.theta0,
            phi_th=phi_th_thin
        )


    def lnu_theta(
        self,
        theta:models.ElectronTemperature,
        phi_th:models.ThermalSynchrotronPhase
    )->models.ThermalSynchrotronLuminosity:
        return models.ThermalSynchrotronLuminosity.from_params(
            mep=self.mep,
            theta=theta,
            theta0=self.grid.theta0,
            eps_th=self.eps_th,
            phi_th=phi_th
        )

    def doppler_delta(
        self,
        beta_sh:FloatArray
    )->FloatArray:
        return quantity_converter.beta_into_doppler_delta(beta_sh)

    def xm_peak(
        self,
        table:DimensionlessThermalPeak,
        tau_th:models.OpticalDepthParameter
    )->NDArray[np.float64]:
        ln_xm_peak = table.ln_xm_peak(
            ln_tau_th=np.log(tau_th.values)
        )
        return np.exp(ln_xm_peak)

    def lambda_peak(
        self,
        table:DimensionlessThermalPeak,
        tau_th:models.OpticalDepthParameter
    )->NDArray[np.float64]:
        ln_lambda_peak = table.ln_lambda_peak(
            ln_tau_th=np.log(tau_th.values)
        )
        return np.exp(ln_lambda_peak)

    def phi_peak(
        self,
        xm_peak:NDArray[np.float64],
        phi_th:models.ThermalSynchrotronPhase
    )->QuantityArray:
        return QuantityArray(
            values = phi_th.values*xm_peak,
            unit = phi_th.unit
        )

    def lnu_peak(
        self,
        table:DimensionlessThermalPeak,
        lnu_th:models.ThermalSynchrotronLuminosity,
        tau_th:models.OpticalDepthParameter
    )->QuantityArray:
        ln_lambda_peak = table.ln_lambda_peak(
            ln_tau_th=np.log(tau_th.values)
        )
        lambda_peak = np.exp(ln_lambda_peak)
        return QuantityArray(
            values = lnu_th.values*lambda_peak,
            unit = lnu_th.unit
        )

    def fnu_peak_observerframe(
        self,
        lnu_peak:QuantityArray,
        doppler_delta:FloatArray
    )->QuantityArray:
        distance = self.distance.quantity
        fnu_peak = doppler_delta**3 * lnu_peak.quantity / (4.0*np.pi*distance**2)
        return QuantityArray.from_Quantity(
            quantity = fnu_peak
        )

    def to_df(
        self,
        table:DimensionlessThermalPeak,
        *,
        a_wind_unit:str = "g cm-1",
        phi_unit:str = "GHz min",
        lnu_unit:str = "erg s-1 Hz-1",
        fnu_unit:str = "mJy",
    )->pd.DataFrame:
        grid = self.grid

        # electron temperature
        theta = self.theta()
        theta_thin_val = self.theta_thin.values

        # shock velocity
        beta_sh=self.beta_sh()
        doppler_delta=self.doppler_delta(beta_sh)

        # wind parameter
        a_syn = self.a_syn(theta)
        a_wind = self.a_wind(a_syn)

        tau_th = self.tau_theta(theta)
        lnu_th = self.lnu_theta(theta,phi_th=grid.phi_th)

        # peak values
        xm_peak = self.xm_peak(table,tau_th)
        phi_peak = self.phi_peak(xm_peak,grid.phi_th)
        lnu_peak = self.lnu_peak(
            table,
            lnu_th=lnu_th,
            tau_th=tau_th
        )
        fnu_peak = self.fnu_peak_observerframe(
            lnu_peak=lnu_peak,
            doppler_delta=doppler_delta
        )

        # optically thin limit
        phi_th_thin = self.phi_th_thin(theta)
        tau_th_thin = self.tau_theta_thinlimit(theta,phi_th_thin)
        lnu_th_thin = self.lnu_theta(
            theta = theta,
            phi_th = phi_th_thin
        )
        xm_peak_thin = self.xm_peak(table,tau_th_thin)
        phi_peak_thin = self.phi_peak(
            xm_peak_thin,
            phi_th_thin
        )
        lnu_peak_thin = self.lnu_peak(
            table=table,
            lnu_th=lnu_th_thin,
            tau_th=tau_th_thin
        )
        fnu_peak_thin = self.fnu_peak_observerframe(
            lnu_peak = lnu_peak_thin,
            doppler_delta=doppler_delta
        )

        df = pd.DataFrame({
            ModelParameterNames.PHI_TH: grid.phi_th.to_value_in(phi_unit).ravel(),
            ModelParameterNames.THETA0: grid.theta0.values.ravel(),
            ModelColumnNames.THETA: theta.values.ravel(),
            ModelColumnNames.THETA_THIN: theta_thin_val.ravel(),
            ModelColumnNames.IS_SSA_THICK: (grid.theta0.values < theta_thin_val).ravel(),
            ModelColumnNames.TAU_THETA: tau_th.values.ravel(),
            ModelColumnNames.LNU_THETA: lnu_th.to_value_in(lnu_unit).ravel(),
            ModelColumnNames.DOPPLER_DELTA: doppler_delta.ravel(),
            ModelColumnNames.PHI_PEAK: phi_peak.to_value_in(phi_unit).ravel(),
            ModelColumnNames.FNU_PEAK: fnu_peak.to_value_in(fnu_unit).ravel(),
            ModelColumnNames.BETA_SH: beta_sh.ravel(),
            ModelColumnNames.A_SYN: a_syn.to_value_in(a_wind_unit).ravel(),
            ModelColumnNames.A_WIND: a_wind.to_value_in(a_wind_unit).ravel(),
            ModelColumnNames.PHI_TH_THIN: phi_th_thin.to_value_in(phi_unit).ravel(),
            ModelColumnNames.TAU_TH_THIN: tau_th_thin.values.ravel(),
            ModelColumnNames.LNU_TH_THIN: lnu_th_thin.to_value_in(lnu_unit).ravel(),
            ModelColumnNames.PHI_PEAK_THIN: phi_peak_thin.to_value_in(phi_unit).ravel(),
            ModelColumnNames.FNU_PEAK_THIN: fnu_peak_thin.to_value_in(fnu_unit).ravel(),
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

    def view_df(self,df:pd.DataFrame,columns:list[str]):
        print(df[columns].to_string(index=False))
