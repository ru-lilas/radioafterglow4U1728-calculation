import numpy as np
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
from typing import Self

from module.mydataclasses import QuantityArray
from module import models
from module.types import FloatArray
from module.inputs_as_dataclass import(
    PhysicalParameters
)
from module.parameter_table import LambdaPeakTable
from enum import StrEnum, auto

class GridColumns(StrEnum):
    A_WIND = auto()
    BETA_SH = auto()
    PHI_PEAK = auto()
    FNU_PEAK = auto()

@dataclass(frozen=True,slots=True)
class ChevalierGrid:
    a_wind_arr: FloatArray
    beta_sh_arr: FloatArray
    phi_peak_arr: FloatArray
    fnu_peak_arr: FloatArray

    @property
    def df(self)->pd.DataFrame:
        df = pd.DataFrame({
            GridColumns.A_WIND: self.a_wind_arr,
            GridColumns.BETA_SH: self.beta_sh_arr,
            GridColumns.PHI_PEAK: self.phi_peak_arr,
            GridColumns.FNU_PEAK: self.fnu_peak_arr
        })
        if df.duplicated(
            subset=[GridColumns.A_WIND,GridColumns.BETA_SH]
        ).any():
            raise ValueError(
                "同じA_windとbeta_shの組が複数存在します."
            )
        else:
            return df

    @property
    def a_wind_axis(self)->FloatArray:
        return np.unique(self.a_wind_arr)

    @property
    def beta_sh_axis(self)->FloatArray:
        return np.unique(self.beta_sh_arr)

    @property
    def meshgrid(
        self,
    ) -> tuple[FloatArray, FloatArray]:

        a_wind_grid, beta_sh_grid = np.meshgrid(
            self.a_wind_axis,
            self.beta_sh_axis,
            indexing="xy",
        )

        return (
            a_wind_grid,
            beta_sh_grid,
        )

    def df_pivot(self,value:str):
        table = (
            self.df.pivot(
                index=GridColumns.BETA_SH,
                columns=GridColumns.A_WIND,
                values=value,
            )
            .reindex(
                index=self.beta_sh_axis,
                columns=self.a_wind_axis,
            )
        )

        if table.isna().any().any():
            raise ValueError(
                f"{value}のパラメータグリッドに欠損があります."
            )

        return table.to_numpy(dtype=np.float64)

    @property
    def phi_peak_grid(self):
        return self.df_pivot(GridColumns.PHI_PEAK)

    @property
    def fnu_peak_grid(self):
        return self.df_pivot(GridColumns.FNU_PEAK)

@dataclass(frozen=True,slots=True)
class ChevalierInputs:
    physical_parameters: PhysicalParameters
    peak_table: LambdaPeakTable

    @classmethod
    def import_from(
        cls,
        path_input: Path,
        path_peak_table: Path
    )->Self:
        return cls(
            physical_parameters = PhysicalParameters.from_yaml(path_input),
            peak_table = LambdaPeakTable.from_csv(path_peak_table)
        )

    @property
    def tau_theta(self)->FloatArray:
        return self.physical_parameters.tau_theta

    @property
    def xm_peak(self)->FloatArray:
        return self.peak_table.interpolated_xm_peak(self.tau_theta)

    @property
    def lambda_theta_peak(self)->FloatArray:
        return self.peak_table.interpolated_lambda_peak(self.tau_theta)

    @property
    def phi_peak(
        self,
    )->QuantityArray:
        phi_theta_arr:FloatArray = self.physical_parameters.phi_theta.value
        unit = self.physical_parameters.phi_theta._unitstr
        return QuantityArray(
            values = phi_theta_arr*self.xm_peak,
            unit = unit
        )

    @property
    def lnu_peak(
        self,
    )->QuantityArray:
        lnu_theta_arr:FloatArray = self.physical_parameters.lnu_theta.value
        unit = self.physical_parameters.lnu_theta._unitstr
        return QuantityArray(
            values = lnu_theta_arr*self.lambda_theta_peak,
            unit = unit
        )

    @property
    def fnu_peak_observerframe(
        self
    )->QuantityArray:
        distance = self.physical_parameters.distance
        doppler_delta = self.physical_parameters.doppler_delta
        fnu_sourceframe = models.lnu_into_fnu(
            lnu = self.lnu_peak,
            distance = distance
        )
        return models.fnu_sourceframe_into_fnu_observerframe(
            fnu_sourceframe=fnu_sourceframe,
            doppler_delta=doppler_delta
        )

    def build_chevalier_grid(
        self,
        a_wind_unit: str,
        phi_unit: str,
        fnu_unit: str
    )->ChevalierGrid:
        return ChevalierGrid(
            a_wind_arr=self.physical_parameters.a_wind.FloatArray_in(a_wind_unit),
            beta_sh_arr = self.physical_parameters.beta_sh,
            phi_peak_arr= self.phi_peak.FloatArray_in(phi_unit),
            fnu_peak_arr= self.fnu_peak_observerframe.FloatArray_in(fnu_unit)
        )
