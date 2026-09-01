from pathlib import Path
from typing import Self, cast
from dataclasses import dataclass
from module.dataframe_processors import convert_ndarray
from module.types import FloatArray
import astropy.units as u
import pandas as pd
import numpy as np
from module import mydataclasses, synchrotron_scaling_values, utils
from module import electron_temperature
from enum import StrEnum, auto
from module import quantity_converter
from module import input_reader
from module.inputs_as_dataclass import PhysicalParameters
from module.mydataclasses import (
    QuantityArray,
    QuantityArrayBase,
    QuantityData,
    YAMLReadable,
    ValueArray
)
from module.utils import FileReader

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
    XM_PEAK = auto()
    LAMBDA_PEAK = auto()
    IDX = auto()

    # def to_df(self):
    #     return pd.DataFrame({
    #         Columns.A_WIND: self.a_wind.values,
    #         Columns.BETA_SH: self.beta_sh,
    #         Columns.EPS_B: self.eps_b,
    #         Columns.EPS_TH: self.eps_th,
    #         Columns.MU: self.mu,
    #         Columns.MU_E: self.mu_e,
    #         Columns.DISTANCE: self.distance.value,
    #         Columns.THETA: self.theta,
    #         Columns.PHI_THETA: self.phi_theta.value,
    #         Columns.TAU_THETA: self.tau_theta,
    #         Columns.LNU_THETA: self.lnu_theta.value,
    #         Columns.DOPPLER_DELTA: self.doppler_delta
    #     })
    #
    # def metadata(self):
    #     return {
    #         Columns.A_WIND: self.a_wind.unit,
    #         Columns.DISTANCE: self.distance.unit,
    #         Columns.PHI_THETA: cast(u.UnitBase,self.phi_theta.unit).to_string(),
    #         Columns.LNU_THETA: cast(u.UnitBase,self.lnu_theta.unit).to_string()
    #     }

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
        df = utils.FileReader.table_from_csv(path)
        return cls(
            tau_theta_ref = convert_ndarray(df,Columns.TAU_THETA),
            xm_peak_ref = convert_ndarray(df,Columns.XM_PEAK),
            lambda_peak_ref = convert_ndarray(df,Columns.LAMBDA_PEAK),
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
