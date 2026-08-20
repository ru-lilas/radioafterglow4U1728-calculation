
from module.compute_lightcurve import BinnedCalculationLightcurve
from module import dataframe_utils, observation
from module.mydataclasses import QuantityArray
from module.types import FloatArray
from module import mystatistics
import numpy as np
import pandas as pd
from enum import StrEnum, auto
from dataclasses import dataclass

class Columns(StrEnum):
    IDX = auto()
    IDX_MIN = auto()
    N_DATA = auto()
    N_PARAM = auto()
    N_DOF = auto()
    CHI2 = auto()
    CHI2MIN = auto()
    CHI2_RED = auto()
    CHI2MIN_RED = auto()
    PVALUE = auto()

def calculate_lightcurve_chi2(
    model: BinnedCalculationLightcurve,
    observed: observation.Lightcurve
):
    t_unit = observed.metadata.t_unit
    fnu_unit = observed.metadata.fnu_unit

    t_model: FloatArray = model.t_center_quantity.FloatArray_in(t_unit)
    fnu_model: FloatArray = model.fnu_averaged_quantity.FloatArray_in(fnu_unit)

    t_observed: FloatArray = observed.to_FloatArray(observation.Columns.T)
    fnu_observed: FloatArray = observed.to_FloatArray(observation.Columns.FNU_NET)
    fnu_err_observed: FloatArray = observed.to_FloatArray(observation.Columns.FNU_NET_ERR)

    if not (
        t_model.size
        == fnu_model.size
        == t_observed.size
        == fnu_observed.size
        == fnu_err_observed.size
    ):
        raise ValueError(
            "理論値と観測値の配列長が一致しません."
        )
    if np.any(~np.isfinite(fnu_model)):
        raise ValueError(
            "理論fluxに非有限値が含まれています."
        )

    if np.any(~np.isfinite(fnu_observed)):
        raise ValueError(
            "観測fluxに非有限値が含まれています."
        )

    if np.any(~np.isfinite(fnu_err_observed)) or np.any(
        fnu_err_observed <= 0.0
    ):
        raise ValueError(
            "観測誤差は有限かつ正でなければなりません."
        )

    return mystatistics.calculate_chi2(
        x_model = fnu_model,
        x_data = fnu_observed,
        sigma = fnu_err_observed
    )

def append_chi2_for_input(
    chi2_records: list[dict[str,int|float]],
    idx: int,
    chi2: float,
):
    return chi2_records.append({
        Columns.IDX: idx,
        Columns.CHI2: chi2
    })

def build_dataframe_chi2(
    chi2_records: list[dict[str,int|float]]
):
    df = pd.DataFrame(chi2_records)
    df[Columns.IDX] = df[Columns.IDX].astype(int)
    df = df.set_index(Columns.IDX)
    return df

def extract_minimum_chi2(
    df_chi2:pd.DataFrame
):
    return dataframe_utils.extract_minimum_row(df_chi2,Columns.CHI2)

@dataclass(frozen=True)
class MinimumChi2Summary:
    idx: int
    chi2_min: float
    n_data: int
    n_param: int

    @property
    def ndof(self)->int:
        return mystatistics.calculate_ndof(
            self.n_data,
            self.n_param
        )
    @property
    def chi2_min_red(self)->float:
        return mystatistics.calculate_reduced_chi2(self.chi2_min,self.ndof)

    @property
    def pvalue(self)->float:
        return mystatistics.calculate_pvalue(self.chi2_min,self.ndof)

    def update_df_attrs(self,df:pd.DataFrame):
        df.attrs[Columns.IDX_MIN] = self.idx
        df.attrs[Columns.CHI2MIN] = self.chi2_min
        df.attrs[Columns.CHI2MIN_RED] = self.chi2_min_red
        df.attrs[Columns.N_DATA] = self.n_data
        df.attrs[Columns.N_PARAM] = self.n_param
        df.attrs[Columns.N_DOF] = self.ndof
        df.attrs[Columns.PVALUE] = self.pvalue
        return

def build_minimum_chi2_summary(
    df_chi2:pd.DataFrame,
    n_data: int,
    n_param: int,
)->MinimumChi2Summary:
    row = extract_minimum_chi2(df_chi2)
    idx_hash = row.name
    if not isinstance(idx_hash, (int, np.integer)):
        raise TypeError(
            "Seriesのindex名は整数でなければなりません."
        )
    else:
        idx = int(idx_hash)

    chi2_value = row[Columns.CHI2]
    if isinstance(chi2_value, bool) or not isinstance(
        chi2_value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(
            "chi2は数値でなければなりません."
        )

    chi2_min = float(chi2_value)

    if not np.isfinite(chi2_min):
        raise ValueError(
            "chi2は有限値でなければなりません."
        )

    return MinimumChi2Summary(
        idx = idx,
        chi2_min= chi2_min,
        n_data = n_data,
        n_param = n_param
    )
