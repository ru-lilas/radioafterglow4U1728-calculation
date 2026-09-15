from dataclasses import dataclass
from os import wait
from pathlib import Path
from typing import Self
from numpy.typing import NDArray
import numpy as np
import pandas as pd
from module import synchrotron_function
from module.utils import FileReader, FileWriter
from enum import StrEnum, auto

class ThermalSynchrotronColumns(StrEnum):
    XM_REF = auto()
    IP_REF = auto()

class ThermalPeakColumns(StrEnum):
    LN_TAU_TH_REF = auto()
    LN_XM_PEAK_REF = auto()
    LN_LAMBDA_PEAK_REF = auto()

@dataclass(frozen=True,slots=True)
class ThermalSynchrotronIntegrals:
    xm_ref:NDArray[np.float64]
    ip_ref:NDArray[np.float64]

    @classmethod
    def build_numerical_table(
        cls,
        xm:NDArray[np.float64]
    )->Self:
        ip = synchrotron_function.thermal_Ip(xm)

        return cls(
            xm_ref=xm,
            ip_ref=ip
        )

    @classmethod
    def from_csv(
        cls,
        path:Path
    )->Self:
        df = FileReader.table_from_csv(path)

        return cls(
            xm_ref = np.asarray(df[ThermalSynchrotronColumns.XM_REF]),
            ip_ref = np.asarray(df[ThermalSynchrotronColumns.IP_REF])
        )

    def to_df(
        self,
    )->pd.DataFrame:
        df = pd.DataFrame({
            ThermalSynchrotronColumns.XM_REF: self.xm_ref,
            ThermalSynchrotronColumns.IP_REF: self.ip_ref
        })
        return df

    def to_csv(
        self,
        path:Path
    )->None:
        FileWriter.df_to_csv(
            path=path,
            df=self.to_df()
        )

    def ip(
        self,
        xm:NDArray[np.float64]
    )->NDArray[np.float64]:
        return np.interp(
            xm,
            self.xm_ref,
            self.ip_ref
        )

@dataclass(frozen=True, slots=True)
class DimensionlessThermalPeak:
    ln_tau_th_ref: NDArray[np.float64]
    ln_xm_peak_ref: NDArray[np.float64]
    ln_lambda_peak_ref: NDArray[np.float64]

    @classmethod
    def build_numerical_table(
        cls,
        ln_tau_th: NDArray[np.float64],
        ln_xm: NDArray[np.float64],
        integral_table: ThermalSynchrotronIntegrals,
    ) -> Self:
        ln_tau_th_ref = np.unique(ln_tau_th)
        ln_xm_ref = np.unique(ln_xm)

        ln_xm_grid, ln_tau_th_grid = np.meshgrid(
            ln_xm_ref,
            ln_tau_th_ref,
            indexing="xy",
        )

        xm_grid = np.exp(ln_xm_grid)
        tau_th_grid = np.exp(ln_tau_th_grid)

        ip = integral_table.ip(xm_grid)
        optical_depth = tau_th_grid * ip / xm_grid

        ln_lambda_syn = (
            2.0 * ln_xm_grid
            + np.log(-np.expm1(-optical_depth))
        )

        peak_indices = np.argmax(ln_lambda_syn, axis=1)
        row_indices = np.arange(ln_tau_th_ref.size)

        return cls(
            ln_tau_th_ref=ln_tau_th_ref,
            ln_xm_peak_ref=ln_xm_grid[
                row_indices,
                peak_indices,
            ],
            ln_lambda_peak_ref=ln_lambda_syn[
                row_indices,
                peak_indices,
            ],
        )

    def to_df(
        self,
    )->pd.DataFrame:
        df = pd.DataFrame({
            ThermalPeakColumns.LN_TAU_TH_REF: self.ln_tau_th_ref,
            ThermalPeakColumns.LN_XM_PEAK_REF: self.ln_xm_peak_ref,
            ThermalPeakColumns.LN_LAMBDA_PEAK_REF: self.ln_lambda_peak_ref,
        })
        return df

    def to_csv(
        self,
        path:Path
    )->None:
        FileWriter.df_to_csv(
            path=path,
            df=self.to_df()
        )

    @classmethod
    def from_csv(
        cls,
        path:Path
    )->Self:
        df = FileReader.table_from_csv(path)

        return cls(
            ln_tau_th_ref = np.asarray(df[ThermalPeakColumns.LN_TAU_TH_REF]),
            ln_xm_peak_ref = np.asarray(df[ThermalPeakColumns.LN_XM_PEAK_REF]),
            ln_lambda_peak_ref = np.asarray(df[ThermalPeakColumns.LN_LAMBDA_PEAK_REF])
        )

    def ln_xm_peak(
        self,
        ln_tau_th:NDArray[np.float64]
    )->NDArray[np.float64]:
        return np.interp(
            ln_tau_th,
            self.ln_tau_th_ref,
            self.ln_xm_peak_ref
        )

    def ln_lambda_peak(
        self,
        ln_tau_th:NDArray[np.float64]
    )->NDArray[np.float64]:
        return np.interp(
            ln_tau_th,
            self.ln_tau_th_ref,
            self.ln_lambda_peak_ref
        )

    def xm_peak(
        self,
        ln_tau_th:NDArray[np.float64]
    )->NDArray[np.float64]:
        ln_xm_peak = self.ln_xm_peak(ln_tau_th)
        return np.exp(ln_xm_peak)

    def lambda_peak(
        self,
        ln_tau_th:NDArray[np.float64]
    )->NDArray[np.float64]:
        ln_lambda = self.ln_lambda_peak(ln_tau_th)
        return np.exp(ln_lambda)
