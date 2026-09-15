from dataclasses import dataclass
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
