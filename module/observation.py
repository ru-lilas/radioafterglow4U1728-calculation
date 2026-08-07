from pathlib import Path
from dataclasses import dataclass
from typing import Self, cast
from dacite import from_dict
from numpy.typing import NDArray
from functools import cached_property
from module.models import ThermalSynchrotron, ThermalSynchrotronTable, calculate_phi, calculate_xm
from module.mydataclasses import QuantityData,QuantityArray
from module.types import FloatArray
from module.utilities import filereaders as fr
import astropy.units as u
import numpy as np
import pandas as pd
from module.types import FloatArray
from enum import StrEnum, auto

class Columns(StrEnum):
    T = auto()
    NU = auto()
    FNU = auto()
    FNU_ERR = auto()

@dataclass
class LightcurveMetadata:
    t_bin: float
    t_unit: str
    fnu_unit: str
    nu_unit: str

    @classmethod
    def from_keyvalue(
            cls,
            path:Path
    )->Self:
        dict_data = fr.read_keyvalue(path)
        return cls(**dict_data)

    @cached_property
    def bin_width(self):
        return u.Quantity(self.t_bin,self.t_unit)

@dataclass
class Lightcurve:
    metadata: LightcurveMetadata
    df: pd.DataFrame

    @property
    def time_bin_bounds(
        self,
    ) -> tuple[float, FloatArray, FloatArray]:
        t: FloatArray = np.asarray(
            self.df[Columns.T],
            dtype=np.float64,
        )

        if t.size < 2:
            raise ValueError(
                "bin幅の推定には2点以上の時刻が必要です."
            )

        dt: FloatArray = np.diff(t)

        if np.any(dt <= 0.0):
            raise ValueError(
                "時刻は単調増加でなければなりません."
            )

        if not np.allclose(dt, dt[0]):
            raise ValueError(
                "観測時刻が等間隔ではありません."
            )

        bin_width = float(dt[0])
        t_left = t - 0.5 * bin_width
        t_right = t + 0.5 * bin_width

        return (bin_width, t_left, t_right)

@dataclass(frozen=True, slots=True)
class LongformatLightcurve:
    metadata: LightcurveMetadata
    df_long: pd.DataFrame

    @classmethod
    def from_csv(
        cls,
        path: Path,
    ) -> Self:
        metadata = LightcurveMetadata.from_keyvalue(path)

        df_long = fr.read_csv(
            path
        )

        required_columns = {
            Columns.T,
            Columns.FNU,
            Columns.FNU_ERR,
            Columns.NU
        }

        missing_columns = required_columns - set(df_long.columns)

        if missing_columns:
            raise ValueError(
                f"必要な列がありません: {sorted(missing_columns)}"
            )

        return cls(
            metadata=metadata,
            df_long=df_long,
        )

    def group_by_frequency(
        self,
    ) -> dict[float, Lightcurve]:
        lightcurves: dict[float, Lightcurve] = {}

        for nu, df_nu in self.df_long.groupby(
            Columns.NU,
            sort=True,
        ):
            nu_value = cast(float,nu)

            lightcurves[nu_value] = Lightcurve(
                metadata=self.metadata,
                df=df_nu.reset_index(drop=True),
            )

        return lightcurves

    def extract_lightcurve(self,nu:float)->Lightcurve:
        lcs = self.group_by_frequency()
        return lcs[nu]
