from collections.abc import Mapping
from pathlib import Path
from dataclasses import asdict, dataclass, fields, replace
from typing import Self, cast, Any, Hashable
from numpy.typing import NDArray
from matplotlib.figure import Figure
from matplotlib.patches import StepPatch
from module import plot_utils, quantity_converter
from module import models,dataframe_utils
from module import mystatistics
from module.models import PhaseParameter, ScenarioParameter, ThermalSynchrotronUtils, calculate_xm
from module.mydataclasses import IntVector, QuantityArray, QuantityArray1D, QuantityData, QuantityGrid,FloatGrid, YAMLReadable
from module.mystatistics import CalculateChiSquared, MinimumChiSquared
from module.plot_utils import LineStyle, MetadataConfig
from module.types import FloatArray, FloatArray1D
import astropy.units as u
import numpy as np
import pandas as pd
from module.utils import DataFrameUtils, FileWriter, Integrator,FileReader
from module import dataframe_processors, observation
from enum import StrEnum, auto
from matplotlib.axes import Axes
from cattrs import Converter
from module import inputs_as_dataclass
from module.models import (
    MetadataNames
)

DATACLASS_CONVERTER = Converter()

class LcColumns(StrEnum):
    T = auto()
    NU = auto()
    FNU_OBSERVER_FRAME = auto()
    FNU_ERR = auto()
    FNU_NET = auto()
    FNU_NET_ERR = auto()
    T_UNIT = auto()
    FNU_UNIT = auto()
    T_CENTER = auto()
    T_LEFT = auto()
    T_RIGHT = auto()
    FNU_AVERAGED = auto()
    FNU_TRANSIENT = auto()
    FNU_PERSISTENT = auto()
    FNU_PERSISTENT_ERR = auto()
    NU_VALUE = auto()
    NU_UNIT = auto()

class ParameterNames(StrEnum):
    A_WIND = auto()
    BETA_SH = auto()
    PHI_PEAK = auto()
    FNU_PEAK = auto()
    PHI_THETA = auto()
    LNU_THETA = auto()
    TAU_THETA = auto()
    EPS_B = auto()
    EPS_TH = auto()
    MU = auto()
    MU_E = auto()
    DISTANCE_VALUE = auto()
    DISTANCE_UNIT = auto()
    PHI_UNIT = auto()
    LNU_UNIT = auto()
    FNU_UNIT = auto()
    A_WIND_UNIT = auto()
    DOPPLER_DELTA = auto()
    CHI2 = auto()

def ensure_string_keys(
    mapping: Mapping[Hashable, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key, value in mapping.items():
        if not isinstance(key, str):
            raise TypeError(
                f"辞書のキーはstrでなければなりません: {key!r}"
            )

        result[key] = value

    return result

@dataclass(frozen=True, slots=True)
class UnitMetadata:
    t_unit: str
    fnu_unit: str

@dataclass(frozen=True, slots=True)
class ChevalierParameters:
    a_wind: QuantityData
    beta_sh: float
    phi_peak: QuantityData
    phi_theta: QuantityData
    lnu_theta: QuantityData
    tau_theta: float
    doppler_delta: float
    eps_b: float
    eps_th: float
    mu: float
    mu_e: float
    distance: QuantityData
    chi2: float

    @classmethod
    def from_dict(
        cls,
        metadata:dict[str,Any]
    )->Self:
        a_wind = QuantityData(
            value = metadata[ParameterNames.A_WIND],
            unit = metadata[ParameterNames.A_WIND_UNIT]
        )
        phi_peak = QuantityData(
            value = metadata[ParameterNames.PHI_PEAK],
            unit = metadata[ParameterNames.PHI_UNIT]
        )
        phi_theta = QuantityData(
            value = metadata[ParameterNames.PHI_THETA],
            unit = metadata[ParameterNames.PHI_UNIT]
        )
        lnu_theta = QuantityData(
            value = metadata[ParameterNames.LNU_THETA],
            unit = metadata[ParameterNames.LNU_UNIT]
        )
        distance = QuantityData(
            value = metadata[ParameterNames.DISTANCE_VALUE],
            unit = metadata[ParameterNames.DISTANCE_UNIT]
        )
        return cls(
            a_wind = a_wind,
            beta_sh = metadata[ParameterNames.BETA_SH],
            phi_peak=phi_peak,
            phi_theta=phi_theta,
            lnu_theta=lnu_theta,
            tau_theta=metadata[ParameterNames.TAU_THETA],
            doppler_delta=metadata[ParameterNames.DOPPLER_DELTA],
            eps_b=metadata[ParameterNames.EPS_B],
            eps_th=metadata[ParameterNames.EPS_TH],
            mu=metadata[ParameterNames.MU],
            mu_e=metadata[ParameterNames.MU_E],
            chi2=metadata[ParameterNames.CHI2],
            distance=distance
        )

    def to_dict(
        self,
        *,
        a_wind_unit: str,
        phi_unit:str,
        lnu_unit:str,
    )->dict[str,float|str]:
        return {
            ParameterNames.A_WIND: self.a_wind.value_in(a_wind_unit),
            ParameterNames.A_WIND_UNIT: a_wind_unit,
            ParameterNames.BETA_SH: self.beta_sh,
            ParameterNames.PHI_PEAK: self.phi_peak.value_in(phi_unit),
            ParameterNames.PHI_THETA: self.phi_theta.value_in(phi_unit),
            ParameterNames.PHI_UNIT: phi_unit,
            ParameterNames.LNU_THETA: self.lnu_theta.value_in(lnu_unit),
            ParameterNames.LNU_UNIT: lnu_unit,
            ParameterNames.TAU_THETA: self.tau_theta,
            ParameterNames.DOPPLER_DELTA: self.doppler_delta,
            ParameterNames.EPS_B: self.eps_b,
            ParameterNames.EPS_TH: self.eps_th,
            ParameterNames.MU: self.mu,
            ParameterNames.MU_E: self.mu_e,
            ParameterNames.DISTANCE_VALUE: self.distance.value,
            ParameterNames.DISTANCE_UNIT: self.distance.unit,
            ParameterNames.CHI2: self.chi2,
        }

# @dataclass(frozen=True,slots=True)
# class LightcurveMetadata:
#     params: ChevalierParameters
#     nu: QuantityData
#
#     @property
#     def as_dict(
#         self,
#     )->dict[str,Any]:
#         params = self.params
#         params_asdict = params.to_dict(
#             a_wind_unit=params.a_wind.unit,
#             phi_unit=params.phi_theta.unit,
#             lnu_unit=params.lnu_theta.unit
#         )
#         nu = self.nu
#         nu_asdict = {
#             LcColumns.NU_VALUE: nu.value,
#             LcColumns.NU_UNIT: nu.unit
#         }
#         return {
#             **params_asdict,
#             **nu_asdict
#         }

@dataclass(frozen=True,slots=True)
class CandidateModelRows:
    model_index: NDArray[np.float64]
    nu: QuantityData

# @dataclass(frozen=True,slots=True)
# class CandidateModelParameterGrid:
#     metadata: CandidateModelRows
#     a_wind: QuantityArray
#     beta_sh: FloatArray
#     phi_peak: QuantityArray
#     fnu_peak: QuantityArray
#     phi_theta: QuantityArray
#     lnu_theta: QuantityArray
#     tau_theta: FloatArray
#     doppler_delta: FloatArray
#     eps_b: float
#     eps_th: float
#     mu: float
#     mu_e: float
#     distance: QuantityData
#
#     def __post_init__(self) -> None:
#         arrays = {
#             ParameterNames.A_WIND: self.a_wind.values,
#             ParameterNames.BETA_SH: self.beta_sh,
#             ParameterNames.PHI_PEAK: self.phi_peak.values,
#             ParameterNames.FNU_PEAK: self.fnu_peak.values,
#             ParameterNames.PHI_THETA: self.phi_theta.values,
#             ParameterNames.LNU_THETA: self.lnu_theta.values,
#             ParameterNames.TAU_THETA: self.tau_theta,
#         }
#
#         for name, array in arrays.items():
#             if array.ndim != 1:
#                 raise ValueError(
#                     f"{name}は1次元でなければなりません. "
#                     f"shape={array.shape}"
#                 )
#
#         sizes = {
#             name: array.size
#             for name, array in arrays.items()
#         }
#
#         if len(set(sizes.values())) != 1:
#             raise ValueError(
#                 "推定パラメータの配列長が一致していません. "
#                 f"{sizes}"
#             )
#
#     @classmethod
#     def from_csv(
#         cls,
#         path: Path
#     )->Self:
#         df = FileReader.table_from_csv(path)
#         metadata = df.attrs
#         return cls(
#             a_wind = QuantityArray(
#                 values = dataframe_utils.extract_column_as_ndarray(
#                     df,
#                     ParameterNames.A_WIND
#                 ),
#                 unit = metadata[ParameterNames.A_WIND_UNIT],
#             ),
#             beta_sh = dataframe_utils.extract_column_as_ndarray(
#                 df,
#                 ParameterNames.BETA_SH
#             ),
#             phi_peak = QuantityArray(
#                 values = dataframe_utils.extract_column_as_ndarray(
#                     df,
#                     ParameterNames.PHI_PEAK
#                 ),
#                 unit = metadata[ParameterNames.PHI_UNIT]
#             ),
#             phi_theta = QuantityArray(
#                 values = dataframe_utils.extract_column_as_ndarray(
#                     df,
#                     ParameterNames.PHI_THETA
#                 ),
#                 unit = metadata[ParameterNames.PHI_UNIT]
#             ),
#             lnu_theta = QuantityArray(
#                 values = dataframe_utils.extract_column_as_ndarray(
#                     df,
#                     ParameterNames.LNU_THETA
#                 ),
#                 unit = metadata[ParameterNames.LNU_UNIT]
#             ),
#             tau_theta = dataframe_utils.extract_column_as_ndarray(
#                     df,
#                     ParameterNames.TAU_THETA
#             ),
#             doppler_delta = dataframe_utils.extract_column_as_ndarray(
#                 df,
#                 ParameterNames.DOPPLER_DELTA
#             ),
#             fnu_peak = QuantityArray(
#                 values = dataframe_utils.extract_column_as_ndarray(
#                     df,
#                     ParameterNames.FNU_PEAK
#                 ),
#                 unit = metadata[ParameterNames.FNU_UNIT]
#             ),
#             eps_b = metadata[ParameterNames.EPS_B],
#             eps_th = metadata[ParameterNames.EPS_TH],
#             mu = metadata[ParameterNames.MU],
#             mu_e = metadata[ParameterNames.MU_E],
#             distance = QuantityData(
#                 value = metadata[ParameterNames.DISTANCE_VALUE],
#                 unit = metadata[ParameterNames.DISTANCE_UNIT],
#             )
#         )
#
#     def to_df(
#         self,
#         phi_unit,
#         fnu_unit,
#         lnu_unit
#     )->pd.DataFrame:
#         a_wind = self.a_wind
#         df = pd.DataFrame({
#             ParameterNames.A_WIND: a_wind.values,
#             ParameterNames.BETA_SH: self.beta_sh,
#             ParameterNames.PHI_PEAK: self.phi_peak.to_value_in(phi_unit),
#             ParameterNames.FNU_PEAK: self.fnu_peak.to_value_in(fnu_unit),
#             ParameterNames.PHI_THETA: self.phi_theta.to_value_in(phi_unit),
#             ParameterNames.LNU_THETA: self.lnu_theta.to_value_in(lnu_unit),
#             ParameterNames.TAU_THETA: self.tau_theta,
#             ParameterNames.DOPPLER_DELTA: self.doppler_delta,
#         })
#         df.attrs[ParameterNames.EPS_B] = self.eps_b
#         df.attrs[ParameterNames.EPS_TH] = self.eps_th
#         df.attrs[ParameterNames.MU] = self.mu
#         df.attrs[ParameterNames.MU_E] = self.mu_e
#         df.attrs[ParameterNames.DISTANCE_VALUE] = self.distance.value
#         df.attrs[ParameterNames.DISTANCE_UNIT] = self.distance.unit
#         df.attrs[ParameterNames.PHI_UNIT] = phi_unit
#         df.attrs[ParameterNames.FNU_UNIT] = fnu_unit
#         df.attrs[ParameterNames.LNU_UNIT] = lnu_unit
#         df.attrs[ParameterNames.A_WIND_UNIT] = a_wind.unit
#         return df
#
#     def extract_bestfit(
#         self,
#         minchi2:MinimumChiSquared
#     )->ChevalierParameters:
#         idx = minchi2.index
#         chi2 = minchi2.value
#         a_wind_arr = self.a_wind
#         phi_peak_arr = self.phi_peak
#         phi_theta_arr = self.phi_theta
#         lnu_theta_arr = self.lnu_theta
#         fnu_peak_arr = self.fnu_peak
#         a_wind = QuantityData(
#             value = a_wind_arr.values[idx],
#             unit = a_wind_arr.unit
#         )
#         phi_peak = QuantityData(
#             value = phi_peak_arr.values[idx],
#             unit = phi_peak_arr.unit
#         )
#         phi_theta = QuantityData(
#             value = phi_theta_arr.values[idx],
#             unit = phi_theta_arr.unit
#         )
#         lnu_theta = QuantityData(
#             value = lnu_theta_arr.values[idx],
#             unit = lnu_theta_arr.unit
#         )
#         fnu_peak = QuantityData(
#             value = fnu_peak_arr.values[idx],
#             unit = fnu_peak_arr.unit
#         )
#         return ChevalierParameters(
#             a_wind = a_wind,
#             beta_sh = self.beta_sh[idx],
#             phi_peak = phi_peak,
#             phi_theta = phi_theta,
#             lnu_theta = lnu_theta,
#             tau_theta = self.tau_theta[idx],
#             doppler_delta= self.doppler_delta[idx],
#             eps_b=self.eps_b,
#             eps_th = self.eps_th,
#             mu = self.mu,
#             mu_e = self.mu_e,
#             distance=self.distance,
#             chi2=chi2
#         )
#
@dataclass(frozen=True,slots=True)
class CandidateModelLightcurveConfig:
    phi: models.PhaseParameter

@dataclass(frozen=True,slots=True)
class CandidateModelInput:
    eps_b: float
    eps_th: float
    mu_e_mu: float
    distance_value: float
    distance_unit: str
    phi_unit: str
    lnu_unit: str
    fnu_unit: str
    a_wind_unit: str
    nu_unit: str

    @classmethod
    def from_df_attrs(
        cls,
        df:pd.DataFrame
    )->Self:
        return cls(
            **ensure_string_keys(df.attrs)
        )

    @property
    def distance(self):
        return QuantityData(
            value = self.distance_value,
            unit = self.distance_unit
        )

@dataclass(frozen=True,slots=True)
class PhaseParameterGrid(QuantityGrid):

    @classmethod
    def build(
        cls,     
        t: QuantityArray1D,
        nu: QuantityArray1D
    )->Self:
        return cls.from_1DArrays(
            row=nu,
            column=t
        )

    def to_xm(
        self,
        phi_th: QuantityArray1D
    )->FloatGrid:
        return self.decompose_by_column(phi_th)

    def to_ln_xm(
        self,
        phi_th: QuantityArray1D
    )->FloatGrid:
        return self.decompose_by_column(phi_th).ln

    def to_log10_xm(
        self,
        phi_th: QuantityArray1D
    )->FloatGrid:
        return self.decompose_by_column(phi_th).log10

@dataclass(frozen=True,slots=True)
class SpecificLuminosityGrid(QuantityGrid):
    @classmethod
    def build(
        cls,
        lambda_grid: FloatGrid,
        lnu_th: QuantityArray1D
    )->Self:
        return cls.from_FloatGrid(
            grid=lambda_grid,
            row_quantity=lnu_th
        )

@dataclass(frozen=True,slots=True)
class ObserverframeSpecificFluxGrid(QuantityGrid):
    def add_bg(
        self,
        fnu_bg:QuantityArray1D
    )->Self:
        q = self.quantity + fnu_bg.quantity
        return type(self).from_quantity(q)

@dataclass(frozen=True,slots=True)
class SourceframeSpecificFluxGrid(QuantityGrid):
    @classmethod
    def from_lnu(
        cls,
        lnu: QuantityGrid,
        distance: QuantityData
    )->Self:
        fnu = quantity_converter.lnu_into_fnu(
            lnu = lnu.quantity,
            distance = distance.quantity
        )
        return cls.from_quantity(fnu)

    def to_observerframe(
        self,
        doppler_factor:QuantityArray1D
    )->ObserverframeSpecificFluxGrid:
        return ObserverframeSpecificFluxGrid(
            values = self.values * (doppler_factor.values[:,None])**3,
            unit = self.unit
        )

@dataclass(frozen=True,slots=True)
class CandidateModelParameters:
    metadata: CandidateModelInput
    model_index: IntVector
    nu: QuantityArray1D
    phi_th: QuantityArray1D
    theta0: QuantityArray1D
    theta:  QuantityArray1D
    tau_th: QuantityArray1D
    lnu_th: QuantityArray1D
    a_wind: QuantityArray1D
    beta_sh: QuantityArray1D
    doppler_delta: QuantityArray1D

    @classmethod
    def from_df(
        cls,
        df:pd.DataFrame
    )->Self:
        metadata = CandidateModelInput.from_df_attrs(df)
        phi_unit = metadata.phi_unit
        fnu_unit = metadata.fnu_unit
        lnu_unit = metadata.lnu_unit
        nu_unit = metadata.nu_unit
        a_wind_unit = metadata.a_wind_unit

        def import_qa_from_the_df(
            column: str,
            unit: str
        )->QuantityArray1D:
            return QuantityArray1D.from_df(df,column,unit)

        return cls(
            metadata = metadata,
            model_index = IntVector.from_df(
                df,
                index_name=models.ColumnNames.MODEL_INDEX
            ),
            nu = import_qa_from_the_df(
                column=models.ColumnNames.NU,
                unit=nu_unit
            ),
            phi_th = import_qa_from_the_df(
                column=models.ColumnNames.PHI_TH,
                unit=phi_unit
            ),
            theta0 = import_qa_from_the_df(
                column=models.ColumnNames.THETA0,
                unit=""
            ),
            theta = import_qa_from_the_df(
                column=models.ColumnNames.THETA,
                unit=""
            ),
            tau_th = import_qa_from_the_df(
                column=models.ColumnNames.TAU_THETA,
                unit=""
            ),
            lnu_th = import_qa_from_the_df(
                column=models.ColumnNames.LNU_THETA,
                unit=lnu_unit
            ),
            a_wind = import_qa_from_the_df(
                column=models.ColumnNames.A_WIND,
                unit=a_wind_unit
            ),
            beta_sh = import_qa_from_the_df(
                column=models.ColumnNames.BETA_SH,
                unit=""
            ),
            doppler_delta = import_qa_from_the_df(
                column=models.ColumnNames.DOPPLER_DELTA,
                unit=""
            ),
        )

@dataclass(frozen=True,slots=True)
class CandidateModelLightcurveGrid:
    metadata: CandidateModelInput
    model_index: IntVector
    nu: QuantityArray1D
    t: QuantityArray1D
    lnu: SpecificLuminosityGrid
    fnu_src_frame: SourceframeSpecificFluxGrid
    fnu_obs_frame: ObserverframeSpecificFluxGrid

    def to_df(self) -> pd.DataFrame:
        metadata = self.metadata
        model_index = self.model_index.values
        t = self.t.quantity
        nu = self.nu.to_value_in(metadata.nu_unit)
        fnu = self.fnu_obs_frame.to_value_in(metadata.fnu_unit)
        lnu = self.lnu.to_value_in(metadata.lnu_unit)
        fnu_src_frame = self.fnu_src_frame.to_value_in(metadata.fnu_unit)

        n_model = model_index.size
        n_time = t.size

        if fnu.shape != (n_model, n_time):
            raise ValueError(
                "fnu shape must be "
                f"({n_model}, {n_time}), "
                f"but got {fnu.shape}"
            )

        candidate_model_index = np.arange(n_model)

        df = pd.DataFrame({
            models.ColumnNames.CANDIDATE_MODEL_INDEX: np.repeat(
                candidate_model_index,
                n_time,
            ),
            models.ColumnNames.NU: np.repeat(
                nu,
                n_time,
            ),
            models.ColumnNames.MODEL_INDEX: np.repeat(
                model_index,
                n_time,
            ),
            models.ColumnNames.T: np.tile(
                t.value,
                n_model,
            ),
            models.ColumnNames.FNU_OBS_FRAME: fnu.ravel(),
            models.ColumnNames.FNU_SRC_FRAME: fnu_src_frame.ravel(),
            models.ColumnNames.LNU: lnu.ravel()
        })
        df.attrs = {
            models.MetadataNames.T_UNIT: self.t.unit,
            **asdict(metadata)
        }

        return df

# @dataclass(frozen=True, slots=True)
# class BinnedBestfitLightcurve:
#     params: ChevalierParameters
#     nu: QuantityData
#     t_center:   QuantityArray
#     t_left:     QuantityArray
#     t_right:    QuantityArray
#     fnu_net:  QuantityArray
#     fnu_persistent: QuantityArray
#     fnu_persistent_err: QuantityArray
#
#     @classmethod
#     def from_csv(
#         cls,
#         path:Path
#     )->Self:
#         df = FileReader.table_from_csv(
#             path=path,
#         )
#         metadata = ensure_string_keys(df.attrs)
#         params = ChevalierParameters.from_dict(metadata)
#         t_unit = metadata[LcColumns.T_UNIT]
#         fnu_unit = metadata[LcColumns.FNU_UNIT]
#         nu = QuantityData(
#             value = metadata[LcColumns.NU_VALUE],
#             unit = metadata[LcColumns.NU_UNIT]
#         )
#         t_center = QuantityArray(
#                 values = dataframe_processors.convert_ndarray(
#                             df,LcColumns.T_CENTER
#                         ),
#                 unit = t_unit
#         )
#         t_right = QuantityArray(
#                 values = dataframe_processors.convert_ndarray(
#                             df,LcColumns.T_RIGHT
#                         ),
#                 unit = t_unit
#         )
#         t_left = QuantityArray(
#                 values = dataframe_processors.convert_ndarray(
#                             df,LcColumns.T_LEFT
#                         ),
#                 unit = t_unit
#         )
#         fnu_net = QuantityArray(
#                 values = dataframe_processors.convert_ndarray(
#                             df,LcColumns.FNU_NET
#                         ),
#                 unit = fnu_unit
#         )
#         fnu_persistent = QuantityArray(
#                 values = dataframe_processors.convert_ndarray(
#                             df,LcColumns.FNU_PERSISTENT
#                         ),
#                 unit = fnu_unit
#         )
#         fnu_persistent_err = QuantityArray(
#                 values = dataframe_processors.convert_ndarray(
#                             df,LcColumns.FNU_PERSISTENT_ERR
#                         ),
#                 unit = fnu_unit
#         )
#         return cls(
#             params=params,
#             nu=nu,
#             t_center=t_center,
#             t_left = t_left,
#             t_right = t_right,
#             fnu_net = fnu_net,
#             fnu_persistent = fnu_persistent,
#             fnu_persistent_err = fnu_persistent_err
#         )
#
#     @property
#     def bin_edges(self):
#         t_unit = self.t_left.unit
#         t_left = self.t_left.to_value_in(t_unit)
#         t_right = self.t_right.to_value_in(t_unit)
#         bin_edges = np.concatenate(
#             (
#                 t_left,
#                 np.array([t_right[-1]]),
#             )
#         )
#         return QuantityArray(
#             values=bin_edges,
#             unit=t_unit
#         )
#
#     @property
#     def fnu_total(
#         self,
#     )->QuantityArray:
#         fnu_net = self.fnu_net
#         fnu_unit = fnu_net.unit
#         fnu_net_arr = fnu_net.to_value_in(fnu_unit)
#         fnu_per = self.fnu_persistent.to_value_in(fnu_unit)
#         fnu_per_err = self.fnu_persistent_err.to_value_in(fnu_unit)
#
#         if not (
#             self.t_center.values.size
#             == fnu_per.size
#             == fnu_per_err.size
#         ):
#             raise ValueError(
#                 "計算光度曲線と定常放射の配列の長さが一致しません."
#             )
#
#         return QuantityArray(
#             values = fnu_net_arr + fnu_per,
#             unit = fnu_unit
#         )
#
#     def to_df(
#         self,
#         *,
#         t_unit:str,
#         fnu_unit:str,
#         a_wind_unit:str,
#         phi_unit:str,
#         lnu_unit:str,
#     )->pd.DataFrame:
#         metadata = self.params.to_dict(
#             a_wind_unit=a_wind_unit,
#             phi_unit=phi_unit,
#             lnu_unit=lnu_unit,
#         )
#         df = pd.DataFrame({
#             LcColumns.T_LEFT: self.t_left.to_value_in(t_unit),
#             LcColumns.T_CENTER: self.t_center.to_value_in(t_unit),
#             LcColumns.T_RIGHT: self.t_right.to_value_in(t_unit),
#             LcColumns.FNU_NET: self.fnu_net.to_value_in(fnu_unit),
#             LcColumns.FNU_PERSISTENT: self.fnu_persistent.to_value_in(fnu_unit),
#             LcColumns.FNU_PERSISTENT_ERR: self.fnu_persistent_err.to_value_in(fnu_unit),
#         })
#         df.attrs = {**metadata}
#         df.attrs[LcColumns.T_UNIT] = t_unit
#         df.attrs[LcColumns.FNU_UNIT] = fnu_unit
#         df.attrs[LcColumns.NU_VALUE] = self.nu.value
#         df.attrs[LcColumns.NU_UNIT] = self.nu.unit
#
#         return df
#
#     def plot_step(
#         self,
#         ax: Axes,
#         style: LineStyle,
#         *,
#         t_unit: str,
#         fnu_unit: str
#     )->StepPatch:
#         fnu_total = self.fnu_total.to_value_in(fnu_unit)
#         bin_edges = self.bin_edges.to_value_in(t_unit)
#
#         return ax.stairs(
#             fnu_total,
#             bin_edges,
#             baseline=self.fnu_persistent.to_value_in(fnu_unit)[0],
#             **style.to_kwargs(),
#         )
#
#     def plot_metadata(
#         self,
#         fig: Figure,
#         string:str
#     ):
#         nu = self.nu
#         params = self.params
#         a_wind_val = params.a_wind.value
#         a_wind_unit = params.a_wind.unit
#         fig.suptitle(
#             string
#         )

@dataclass(frozen=True,slots=True)
class CandidateModelLightcurveMetadata(CandidateModelInput):
    t_unit: str

@dataclass(frozen=True,slots=True)
class CandidateModelBinnedLightcurve:
    metadata: CandidateModelLightcurveMetadata
    candidate_model_index: int
    model_index: int
    nu: QuantityData
    fnu_flare: QuantityArray1D
    t_center: QuantityArray1D
    t_left: QuantityArray1D
    t_right: QuantityArray1D

@dataclass(frozen=True,slots=True)
class CandidateModelLightcurve:
    metadata: CandidateModelLightcurveMetadata
    candidate_model_index: int
    model_index: int
    nu: QuantityData
    t: QuantityArray1D
    fnu_obs_frame: QuantityArray1D

    @classmethod
    def from_df(
        cls,
        candidate_model_index: int,
        df:pd.DataFrame
    )->Self:
        metadata = CandidateModelLightcurveMetadata.from_df_attrs(df)
        nu_unit = metadata.nu_unit
        nu_array = QuantityArray1D.from_df(
            df,models.ColumnNames.NU,metadata.nu_unit
        )
        model_index = IntVector.from_df(
            df,models.ColumnNames.MODEL_INDEX
        ).as_scalar()

        return cls(
            metadata = metadata,
            candidate_model_index=candidate_model_index,
            model_index = model_index,
            t = QuantityArray1D.from_df(
                df,models.ColumnNames.T,metadata.t_unit
            ),
            nu = nu_array.index(0),
            fnu_obs_frame = QuantityArray1D.from_df(
                df,models.ColumnNames.FNU_OBS_FRAME,metadata.fnu_unit
            ),
        )

    @staticmethod
    def _make_bin_edges(
        t_obs_value: FloatArray,
        width: float,
        drop_incomplete_bin: bool,
    )-> FloatArray:
        start = t_obs_value[0]
        stop = t_obs_value[-1]
        duration = t_obs_value[-1] - t_obs_value[0]

        n_bin = int(np.floor(duration/(width)))

        t_edge:FloatArray = (
            t_obs_value[0]
            + width * np.arange(n_bin + 1, dtype=np.float64)
        )

        if (
                not drop_incomplete_bin
                and not np.isclose(t_edge[-1],t_obs_value[-1])
        ):
            t_edge = np.append(t_edge, t_obs_value[-1])
        else:
            pass
        return t_edge

    @staticmethod
    def validate_time_coverage(
        t_model: QuantityArray1D,
        binning: observation.Binning,
    ) -> None:
        t_model_arr = t_model.values

        if t_model_arr.size < 2:
            raise ValueError(
                "理論光度曲線には2点以上の時刻が必要です."
            )

        if (
            binning.t_left[0] < t_model_arr[0]
            and not np.isclose(
                binning.t_left[0],
                t_model_arr[0],
            )
        ):
            raise ValueError(
                "理論光度曲線が最初の観測binを覆っていません."
            )

        if (
            binning.t_right[-1] > t_model_arr[-1]
            and not np.isclose(
                binning.t_right[-1],
                t_model_arr[-1],
            )
        ):
            raise ValueError(
                "理論光度曲線が最後の観測binを覆っていません."
            )
    def insert_zero_boundary(
        self
    )->Self:
        """
            配列の左端に時刻ゼロでの境界条件を挿入する
        """
        t_values = self.t.values
        fnu_values = self.fnu_obs_frame.values

        if t_values[0] == 0.0:
            return self

        if t_values[0] < 0.0:
            raise ValueError(
                "計算光度曲線に負の時刻が含まれています."
                f"t_min={t_values[0]}"
            )

        return type(self)(
            metadata=self.metadata,
            candidate_model_index=self.candidate_model_index,
            model_index = self.model_index,
            t=QuantityArray1D(
                values=np.concatenate((
                    np.array([0.0], dtype=np.float64),
                    t_values,
                )),
                unit=self.t.unit,
            ),
            fnu_obs_frame=QuantityArray1D(
                values=np.concatenate((
                    np.array([0.0], dtype=np.float64),
                    fnu_values,
                )),
                unit=self.fnu_obs_frame.unit,
            ),
            nu=self.nu,
        )

    def bin_average(
        self,
        bin_config: observation.Binning,
        drop_incomplete_bin: bool = True
    )->CandidateModelBinnedLightcurve:

        t_unit = bin_config.t_unit
        fnu_unit = self.fnu_obs_frame.unit

        ## 計算光度曲線が観測の時間範囲を覆っているかcheck
        t = self.t
        fnu = self.fnu_obs_frame
        self.validate_time_coverage(t,bin_config)

        width = bin_config.bin_width
        t_model = t.to_value_in(t_unit)
        fnu_model = fnu.to_value_in(fnu_unit)

        if np.any(np.diff(t_model) <= 0.0):
            raise ValueError(
                "計算データの時刻配列が単調増加でありません."
            )


        t_center = bin_config.t_center
        t_left = bin_config.t_left
        t_right = bin_config.t_right
        num_bins = t_center.size

        if not (
            t_left.size
            == t_right.size
            == num_bins
        ):
            raise ValueError(
                "観測binの配列長が一致しません."
            )

         # 計算時間範囲に完全に含まれるbin
        covered = (
            (t_left >= t_model[0])
            & (t_right <= t_model[-1])
        )

        if drop_incomplete_bin:
            t_center = t_center[covered]
            t_left = t_left[covered]
            t_right = t_right[covered]

            if t_center.size == 0:
                raise ValueError(
                    "計算時間範囲に完全に含まれるbinがありません."
                )

        def interpolate_flux(
            t_evaluate: FloatArray,
        ) -> FloatArray:
            return np.interp(
                t_evaluate,
                t_model,
                fnu_model,
            )

        # shapeはいずれも(M, B)
        fnu_left = interpolate_flux(t_left)
        fnu_right = interpolate_flux(t_right)

        # shapeはいずれも(M, B)
        fnu_left = interpolate_flux(t_left)
        fnu_right = interpolate_flux(t_right)

        fnu_average_by_bin: list[float] = []

        for i, (left, right) in enumerate(
            zip(t_left, t_right, strict=True)
        ):
            inside = (
                (t_model > left)
                & (t_model < right)
            )

            t_bin = np.concatenate((
                np.array([left], dtype=np.float64),
                t_model[inside],
                np.array([right], dtype=np.float64),
            ))

            fnu_bin = np.concatenate((
                np.array([fnu_left[i]], dtype=np.float64),
                fnu_model[inside],
                np.array([fnu_right[i]], dtype=np.float64),
            ))

            flux_integral = np.trapezoid(
                fnu_bin,
                x=t_bin,
            )

            fnu_average_by_bin.append(
                float(flux_integral / (right - left))
            )

        fnu_average = QuantityArray1D(
            values=np.asarray(
                fnu_average_by_bin,
                dtype=np.float64,
            ),
            unit=fnu_unit,
        )

        return CandidateModelBinnedLightcurve(
            metadata=self.metadata,
            candidate_model_index=self.candidate_model_index,
            model_index= self.model_index,
            nu = self.nu,
            t_center=QuantityArray1D(t_center,t_unit),
            t_left=QuantityArray1D(t_left,t_unit),
            t_right=QuantityArray1D(t_right,t_unit),
            fnu_flare=fnu_average,
        )


@dataclass(frozen=True, slots=True)
class CandidateModelTotalLightcurve:
    metadata: CandidateModelLightcurveMetadata
    candidate_model_index: int
    model_index: int
    nu: QuantityData
    t_center: QuantityArray1D
    t_left: QuantityArray1D
    t_right: QuantityArray1D
    fnu_flare: QuantityArray1D
    fnu_bg: QuantityArray1D
    fnu_total: QuantityArray1D

    @classmethod
    def from_df(
        cls,
        candidate_model_index:int,
        metadata:CandidateModelLightcurveMetadata,
        df:pd.DataFrame
    ):
        nu_arr = QuantityArray1D.from_df(
            df,models.ColumnNames.NU,metadata.nu_unit
        )
        model_index = IntVector.from_df(
            df,models.ColumnNames.MODEL_INDEX
        ).as_scalar()
        return cls(
            metadata = metadata,
            candidate_model_index=candidate_model_index,
            model_index = model_index,
            t_center = QuantityArray1D.from_df(
                df,models.ColumnNames.T_CENTER,metadata.t_unit
            ),
            t_left = QuantityArray1D.from_df(
                df,models.ColumnNames.T_LEFT,metadata.t_unit
            ),
            t_right = QuantityArray1D.from_df(
                df,models.ColumnNames.T_RIGHT,metadata.t_unit
            ),
            nu = nu_arr.index(0),
            fnu_flare = QuantityArray1D.from_df(
                df,models.ColumnNames.FNU_FLARE,metadata.fnu_unit
            ),
            fnu_bg = QuantityArray1D.from_df(
                df,models.ColumnNames.FNU_BG,metadata.fnu_unit
            ),
            fnu_total = QuantityArray1D.from_df(
                df,models.ColumnNames.FNU_TOTAL,metadata.fnu_unit
            ),
        )

    @classmethod
    def from_binned_lightcurve(
        cls,
        lightcurve: CandidateModelBinnedLightcurve,
        fnu_bg: QuantityArray1D,
    ) -> Self:
        if lightcurve.t_center.length != fnu_bg.length:
            raise ValueError(
                "光度曲線とbackgroundの配列長が"
                "一致していません."
            )

        fnu_total = QuantityArray1D.from_quantity(
            lightcurve.fnu_flare.quantity
            + fnu_bg.quantity
        )

        return cls(
            metadata=lightcurve.metadata,
            candidate_model_index=(
                lightcurve.candidate_model_index
            ),
            model_index = lightcurve.model_index,
            nu=lightcurve.nu,
            t_center=lightcurve.t_center,
            t_left=lightcurve.t_left,
            t_right=lightcurve.t_right,
            fnu_flare=lightcurve.fnu_flare,
            fnu_bg=fnu_bg,
            fnu_total=fnu_total,
        )

    @property
    def bin_edges(self):
        t_unit = self.t_left.unit
        t_left = self.t_left.to_value_in(t_unit)
        t_right = self.t_right.to_value_in(t_unit)
        bin_edges = np.concatenate(
            (
                t_left,
                np.array([t_right[-1]]),
            )
        )
        return QuantityArray(
            values=bin_edges,
            unit=t_unit
        )

    def plot(
        self,
        plotter: plot_utils.Plotter,
        style: LineStyle,
        *,
        t_unit: str,
        fnu_unit: str
    )->StepPatch:
        bin_edges_q = self.bin_edges
        return plotter.step(
            values=self.fnu_total.to_value_in(fnu_unit),
            edges=bin_edges_q.to_value_in(t_unit),
            style=style,
            baseline=self.fnu_bg.to_value_in(fnu_unit)[0]
        )

@dataclass(frozen=True, slots=True)
class CandidateModelFitResult:
    lightcurve: CandidateModelTotalLightcurve
    chi2: float
    ndof: int

    @classmethod
    def from_binned_lightcurve(
        cls,
        lightcurve: CandidateModelTotalLightcurve,
        fnu_data: QuantityArray1D,
        fnu_err: QuantityArray1D,
        *,
        n_params: int
    )->Self:
        n_data = lightcurve.fnu_total.length
        fnu_unit = lightcurve.metadata.fnu_unit

        if not (
            fnu_data.length
            == fnu_err.length
            == n_data
        ):
            raise ValueError(
            "モデル, 観測値, 観測誤差の"
            "配列長が一致していません."
        )
        chi2 = CalculateChiSquared.from_vector(
            model = lightcurve.fnu_total.to_floatarray1d(fnu_unit),
            observed = fnu_data.to_floatarray1d(fnu_unit),
            error = fnu_err.to_floatarray1d(fnu_unit)
        )

        return cls(
            lightcurve=lightcurve,
            chi2=chi2,
            ndof=n_data-n_params
        )

    def to_df(
        self,
    )->pd.DataFrame:
        metadata = self.lightcurve.metadata
        n_bin = self.lightcurve.t_center.length

        t_unit = metadata.t_unit
        nu_unit = metadata.nu_unit
        fnu_unit = metadata.fnu_unit

        if not (
            self.lightcurve.t_left.length
            == self.lightcurve.t_right.length
            == self.lightcurve.fnu_total.length
            == n_bin
        ):
            raise ValueError(
                "bin平均光度曲線の配列長が一致しません."
            )

        df = pd.DataFrame({
            models.ColumnNames.CANDIDATE_MODEL_INDEX: np.full(
                n_bin,
                self.lightcurve.candidate_model_index,
                dtype=np.int64
            ),
            models.ColumnNames.MODEL_INDEX: np.full(
                n_bin,
                self.lightcurve.model_index,
                dtype=np.int64
            ),
            models.ColumnNames.NU: np.full(
                n_bin,
                self.lightcurve.nu.value_in(nu_unit),
                dtype=np.float64
            ),
            models.ColumnNames.T_CENTER: self.lightcurve.t_center.to_value_in(t_unit),
            models.ColumnNames.T_LEFT: self.lightcurve.t_left.to_value_in(t_unit),
            models.ColumnNames.T_RIGHT: self.lightcurve.t_right.to_value_in(t_unit),
            models.ColumnNames.FNU_FLARE: self.lightcurve.fnu_flare.to_value_in(fnu_unit),
            models.ColumnNames.FNU_BG: self.lightcurve.fnu_bg.to_value_in(fnu_unit),
            models.ColumnNames.FNU_TOTAL: self.lightcurve.fnu_total.to_value_in(fnu_unit),
            models.ChiSquaredParameters.CHI_SQUARED: self.chi2,
            models.ChiSquaredParameters.N_DOF: self.ndof,
        })
        df.attrs.update({**asdict(metadata)})
        return df

@dataclass(frozen=True, slots=True)
class CandidateModelParameter(CandidateModelInput):
    a_wind_value: float
    beta_sh: float

@dataclass(frozen=True, slots=True)
class CandidateModelFitResults:
    metadata: CandidateModelLightcurveMetadata
    models: tuple[CandidateModelFitResult,...]

    @classmethod
    def from_df_long(
        cls,
        df_long:pd.DataFrame
    )->Self:
        metadata = CandidateModelLightcurveMetadata.from_df_attrs(df_long)
        lightcurves = []
        for cand_idx,df in df_long.groupby(
                models.ColumnNames.CANDIDATE_MODEL_INDEX
            ):
            if not isinstance(cand_idx,int):
                raise TypeError(
                    "indexの型が不正です."
                )

            lc = CandidateModelTotalLightcurve.from_df(
                candidate_model_index=cand_idx,
                metadata=metadata,
                df=df
            )
            chi2_arr = QuantityArray1D.from_df(
                df,models.ChiSquaredParameters.CHI_SQUARED,""
            )
            ndof_arr = QuantityArray1D.from_df(
                df,models.ChiSquaredParameters.N_DOF,""
            )
            lightcurves.append(
                CandidateModelFitResult(
                    lightcurve = lc,
                    chi2 = chi2_arr.values[0],
                    ndof = ndof_arr.values[0]
                )
            )
        return cls(
            metadata=metadata,
            models=tuple(lightcurves)
        )

    @property
    def n_candidates(self):
        return len(self.models)

@dataclass(frozen=True, slots=True)
class CandidateModelMetadata(models.ScenarioParameter):
    a_wind: QuantityData
    beta_sh: QuantityData
    nu: QuantityData

    @classmethod
    def from_df(
        cls,
        df_models:pd.DataFrame,
        model_index:int
    ):
        row:pd.Series = df_models.loc[
            df_models[
                models.ColumnNames.MODEL_INDEX
            ] == model_index
        ].iloc[0]
        return cls(
            eps_b=QuantityData.from_attrs(
                df_models,
                key=models.MetadataNames.EPS_B,
            ),
            eps_th=QuantityData.from_attrs(
                df_models,
                key=models.MetadataNames.EPS_TH,
            ),
            mu_e_mu=QuantityData.from_attrs(
                df_models,
                key=models.MetadataNames.MU_E_MU,
            ),
            distance=QuantityData.from_attrs(
                df_models,
                key=models.MetadataNames.DISTANCE_VALUE,
                unit_key=models.MetadataNames.DISTANCE_UNIT
            ),
            a_wind = QuantityData.from_series(
                row=row,
                column=models.ColumnNames.A_WIND,
                unit=df_models.attrs[ParameterNames.A_WIND_UNIT]
            ),
            beta_sh = QuantityData.from_series(
                row=row,
                column=models.ColumnNames.BETA_SH,
                unit=""
            ),
            nu = QuantityData.from_series(
                row=row,
                column=models.ColumnNames.NU,
                unit=df_models.attrs[models.MetadataNames.NU_UNIT]
            ),
        )



# @dataclass(frozen=True,slots=True)
# class BinnedCalculationLightcurve:
#     estimated_params: CandidateModelParameterGrid
#     nu: QuantityData
#     units: UnitMetadata
#     t_center: FloatArray
#     t_left: FloatArray
#     t_right: FloatArray
#     fnu: QuantityArray
#
#     @property
#     def bin_edges(self):
#         return np.concatenate(
#             (
#                 self.t_left,
#                 np.array([self.t_right[-1]]),
#             )
#         )
#
#     @property
#     def bin_edges_quantity(self):
#         return QuantityArray(
#             values=self.bin_edges,
#             unit=self.units.t_unit
#         )
#
#     @property
#     def t_left_quantity(self)->QuantityArray:
#         return QuantityArray(
#             values = self.t_left,
#             unit = self.units.t_unit
#         )
#
#     @property
#     def t_right_quantity(self)->QuantityArray:
#         return QuantityArray(
#             values = self.t_right,
#             unit = self.units.t_unit
#         )
#
#     @property
#     def t_center_quantity(self)->QuantityArray:
#         return QuantityArray(
#             values = self.t_center,
#             unit = self.units.t_unit
#         )
#
#     def fnu_total(
#         self,
#         persistent: QuantityArray,
#         persistent_err: QuantityArray
#     )->QuantityArray:
#         fnu_unit = self.units.fnu_unit
#         fnu_net = self.fnu.to_value_in(fnu_unit)
#         fnu_per = persistent.to_value_in(fnu_unit)
#         fnu_per_err = persistent_err.to_value_in(fnu_unit)
#
#         if not (
#             self.t_center.size
#             == fnu_per.size
#             == fnu_per_err.size
#         ):
#             raise ValueError(
#                 "計算光度曲線と定常放射の配列の長さが一致しません."
#             )
#
#         return QuantityArray(
#             values = fnu_net + fnu_per,
#             unit = fnu_unit
#         )
#
#     def extract(
#         self,
#         minchi2: MinimumChiSquared,
#         persistent: QuantityArray,
#         persistent_err: QuantityArray
#     )->BinnedBestfitLightcurve:
#         params_best = self.estimated_params.extract_bestfit(minchi2)
#         idx = minchi2.index
#         fnu_best = QuantityArray(
#             values = self.fnu.values[idx],
#             unit = self.fnu.unit
#         )
#         return BinnedBestfitLightcurve(
#             params = params_best,
#             nu=self.nu,
#             fnu_net=fnu_best,
#             fnu_persistent=persistent,
#             fnu_persistent_err=persistent_err,
#             t_center=self.t_center_quantity,
#             t_right=self.t_right_quantity,
#             t_left=self.t_left_quantity
#         )
#
#
#     @classmethod
#     def from_csv(
#         cls,
#         path:Path
#     )->Self:
#         df = FileReader.table_from_csv(
#             path=path,
#         )
#         metadata = ensure_string_keys(df.attrs)
#         params = ChevalierParameters.from_dict(metadata)
#         t_unit = metadata[LcColumns.T_UNIT]
#         fnu_unit = metadata[LcColumns.FNU_UNIT]
#         nu = QuantityData(
#             value = metadata[LcColumns.NU_VALUE],
#             unit = metadata[LcColumns.NU_UNIT]
#         )
#         fnu_net = QuantityArray1D(
#                 values = dataframe_processors.convert_ndarray(
#                             df,LcColumns.FNU_NET
#                         ),
#                 unit = fnu_unit
#         )
#         t = QuantityArray1D(
#                 values = dataframe_processors.convert_ndarray(
#                             df,LcColumns.T
#                         ),
#                 unit = t_unit
#         )
#         return cls(
#             metadata=LightcurveMetadata(
#                 params = params,
#                 nu = nu
#             ),
#             t = t,
#             fnu=fnu_net
#         )
#
#     def to_df(
#         self
#     )->pd.DataFrame:
#         t = self.t
#         fnu = self.fnu
#         df = pd.DataFrame({
#             LcColumns.T: t.values,
#             LcColumns.FNU_NET: fnu.values
#         })
#         df.attrs[LcColumns.T_UNIT] = t.unit
#         df.attrs[LcColumns.FNU_UNIT] = fnu.unit
#         df.attrs.update(self.metadata.as_dict)
#         return df
#
#     def dump(
#         self,
#         path: Path
#     )->None:
#         df = self.to_df()
#         return FileWriter.df_to_csv(
#             path=path,
#             df=df,
#             save_attrs=True
#         )
#
#     def fnu_total(
#         self,
#         fnu_bg:QuantityData
#     )->QuantityArray1D:
#         fnu_net = self.fnu
#         fnu_unit = fnu_net.unit
#         return QuantityArray1D(
#             values = fnu_net.to_value_in(fnu_unit) + fnu_bg.value_in(fnu_unit),
#             unit = fnu_unit
#         )
#
#     def plot(
#         self,
#         ax:Axes,
#         style:plot_utils.LineStyle,
#         *,
#         fnu_bg: QuantityData,
#         t_unit: str,
#         fnu_unit: str
#     ):
#         t = self.t.to_value_in(t_unit)
#         fnu_tot = self.fnu_total(fnu_bg=fnu_bg).to_value_in(fnu_unit)
#         ax.plot(
#             t,fnu_tot,
#             **style.to_kwargs()
#         )
#
# @dataclass(frozen=True, slots=True)
# class CalculationLightcurves:
#     params: CandidateModelParameterGrid
#     nu: QuantityData
#     t: QuantityArray
#     fnu: QuantityArray
#
#     def __post_init__(self)->None:
#         t_arr = self.t.values
#         fnu_grid = self.fnu.values
#
#         if t_arr.ndim != 1:
#             raise ValueError(
#                 "tは1次元配列でなければなりません. "
#                 f"shape={t_arr.shape}"
#             )
#
#         if fnu_grid.ndim != 2:
#             raise ValueError(
#                 "fnuは2次元配列でなければなりません. "
#                 f"shape={fnu_grid.shape}"
#             )
#
#         num_times = t_arr.size
#         num_flux_times = fnu_grid.shape[1]
#
#         if num_times != num_flux_times:
#             raise ValueError(
#                 "tの要素数とfnuの列数が一致していません. "
#                 f"t={t_arr.shape}, "
#                 f"fnu={fnu_grid.shape}"
#             )
#
#         return
#
#     @staticmethod
#     def _make_bin_edges(
#         t_obs_value: FloatArray,
#         width: float,
#         drop_incomplete_bin: bool,
#     )-> FloatArray:
#         start = t_obs_value[0]
#         stop = t_obs_value[-1]
#         duration = t_obs_value[-1] - t_obs_value[0]
#
#         n_bin = int(np.floor(duration/(width)))
#
#         t_edge:FloatArray = (
#             t_obs_value[0]
#             + width * np.arange(n_bin + 1, dtype=np.float64)
#         )
#
#         if (
#                 not drop_incomplete_bin
#                 and not np.isclose(t_edge[-1],t_obs_value[-1])
#         ):
#             t_edge = np.append(t_edge, t_obs_value[-1])
#         else:
#             pass
#         return t_edge
#
#     @staticmethod
#     def validate_time_coverage(
#         t_model: QuantityArray,
#         binning: observation.Binning,
#     ) -> None:
#         t_model_arr = t_model.values
#
#         if t_model_arr.size < 2:
#             raise ValueError(
#                 "理論光度曲線には2点以上の時刻が必要です."
#             )
#
#         if (
#             binning.t_left[0] < t_model_arr[0]
#             and not np.isclose(
#                 binning.t_left[0],
#                 t_model_arr[0],
#             )
#         ):
#             raise ValueError(
#                 "理論光度曲線が最初の観測binを覆っていません."
#             )
#
#         if (
#             binning.t_right[-1] > t_model_arr[-1]
#             and not np.isclose(
#                 binning.t_right[-1],
#                 t_model_arr[-1],
#             )
#         ):
#             raise ValueError(
#                 "理論光度曲線が最後の観測binを覆っていません."
#             )
#     def with_zero_boundary(
#         self
#     )->Self:
#         """
#             配列の左端に時刻ゼロでの境界条件を挿入する
#         """
#         t_values = self.t.values
#         fnu_values = self.fnu.values
#
#         if t_values[0] == 0.0:
#             return self
#
#         if t_values[0] < 0.0:
#             raise ValueError(
#                 "計算光度曲線に負の時刻が含まれています."
#                 f"t_min={t_values[0]}"
#             )
#
#
#         return type(self)(
#             params=self.params,
#             t=QuantityArray(
#                 values=np.concatenate((
#                     np.array([0.0], dtype=np.float64),
#                     t_values,
#                 )),
#                 unit=self.t.unit,
#             ),
#             fnu=QuantityArray(
#                 values=np.concatenate((
#                     np.zeros(
#                         (fnu_values.shape[0], 1),
#                         dtype=np.float64,
#                     ),
#                     fnu_values,
#                 ), axis=1),
#                 unit=self.fnu.unit,
#             ),
#             nu=self.nu,
#         )
#
#     def bin_average(
#         self,
#         bin_config: observation.Binning,
#         drop_incomplete_bin: bool = True
#     )->BinnedCalculationLightcurve:
#
#         ## 計算光度曲線が観測の時間範囲を覆っているかcheck
#         t = self.t
#         fnu = self.fnu
#         self.validate_time_coverage(t,bin_config)
#
#         t_unit = bin_config.t_unit
#         fnu_unit = fnu.unit
#
#         width = bin_config.bin_width
#         t_model = t.to_value_in(t_unit)
#         fnu_model = fnu.to_value_in(fnu_unit)
#
#         if np.any(np.diff(t_model) <= 0.0):
#             raise ValueError(
#                 "計算データの時刻配列が単調増加でありません."
#             )
#
#
#         t_center = bin_config.t_center
#         t_left = bin_config.t_left
#         t_right = bin_config.t_right
#         num_bins = t_center.size
#
#         if not (
#             t_left.size
#             == t_right.size
#             == num_bins
#         ):
#             raise ValueError(
#                 "観測binの配列長が一致しません."
#             )
#
#          # 計算時間範囲に完全に含まれるbin
#         covered = (
#             (t_left >= t_model[0])
#             & (t_right <= t_model[-1])
#         )
#
#         if drop_incomplete_bin:
#             t_center = t_center[covered]
#             t_left = t_left[covered]
#             t_right = t_right[covered]
#
#             if t_center.size == 0:
#                 raise ValueError(
#                     "計算時間範囲に完全に含まれるbinがありません."
#                 )
#
#         def interpolate_flux(
#             t_evaluate: FloatArray,
#         ) -> FloatGrid:
#             """
#             全モデルについて、指定時刻で線形補間する
#
#             Returns
#             -------
#             shape: (M, len(t_evaluate))
#             """
#             right_index = np.searchsorted(
#                 t_model,
#                 t_evaluate,
#                 side="left",
#             )
#
#             # t_modelの範囲内であることは上で検証済み
#             right_index = np.clip(
#                 right_index,
#                 1,
#                 t_model.size - 1,
#             )
#             left_index = right_index - 1
#
#             t0 = t_model[left_index]
#             t1 = t_model[right_index]
#
#             fraction = (
#                 (t_evaluate - t0)
#                 / (t1 - t0)
#             )
#
#             fnu0 = fnu_model[:, left_index]
#             fnu1 = fnu_model[:, right_index]
#
#             return (
#                 fnu0
#                 + (fnu1 - fnu0)
#                 * fraction[np.newaxis, :]
#             )
#
#         # shapeはいずれも(M, B)
#         fnu_left = interpolate_flux(t_left)
#         fnu_right = interpolate_flux(t_right)
#
#         fnu_average_by_bin: list[FloatArray] = []
#
#         for i, (left, right) in enumerate(
#             zip(t_left, t_right, strict=True)
#         ):
#             inside = (
#                 (t_model > left)
#                 & (t_model < right)
#             )
#
#             # shape: (K_i,)
#             t_bin = np.concatenate((
#                 np.array([left], dtype=np.float64),
#                 t_model[inside],
#                 np.array([right], dtype=np.float64),
#             ))
#
#             # shape: (M, K_i)
#             fnu_bin = np.concatenate(
#                 (
#                     fnu_left[:, i:i + 1],
#                     fnu_model[:, inside],
#                     fnu_right[:, i:i + 1],
#                 ),
#                 axis=1,
#             )
#
#             # 時間軸axis=1について積分
#             # shape: (M,)
#             flux_integral = np.trapezoid(
#                 fnu_bin,
#                 x=t_bin,
#                 axis=1,
#             )
#
#             fnu_average_by_bin.append(
#                 flux_integral / (right - left)
#             )
#
#         # (B個の(M,)配列) -> (M, B)
#         fnu_average = QuantityArray(
#             values = np.stack(
#                 fnu_average_by_bin,
#                 axis=1,
#             ),
#             unit = fnu_unit
#         )
#
#         return BinnedCalculationLightcurve(
#             estimated_params=self.params,
#             units=UnitMetadata(
#                 t_unit=t_unit,
#                 fnu_unit=str(fnu_unit),
#             ),
#             nu = self.nu,
#             t_center=t_center,
#             t_left=t_left,
#             t_right=t_right,
#             fnu=fnu_average,
#         )
#
#     def extract_bestfit_lc(
#         self,
#         minchi2:MinimumChiSquared
#     )->CalculationLightcurve:
#         params = self.params.extract_bestfit(minchi2)
#         idx = minchi2.index
#         t = QuantityArray1D(
#             values = self.t.values,
#             unit = self.t.unit
#         )
#         fnu_best = QuantityArray1D(
#             values = self.fnu.values[idx],
#             unit = self.fnu.unit
#         )
#         return CalculationLightcurve(
#             metadata = LightcurveMetadata(
#                 params = params,
#                 nu = self.nu
#             ),
#             t = t,
#             fnu = fnu_best
#         )

# def compute(
#     config: inputs_as_dataclass.ModelConfigure,
#     params: CandidateModelParameterGrid,
#     integral_table : models.ThermalSynchrotronTable
# )->CalculationLightcurves:
#     t = config.time.quantity_array
#     nu = QuantityData.from_Quantity(config.nu.quantity)
#     phi = models.PhaseParameter.from_time_given_nu(
#         t = t,
#         nu = nu
#     )
#     xm = models.DimensionlessFrequency.from_phi(
#         phi = phi,
#         phi_theta = models.PhaseParameter(
#             values = params.phi_theta.values,
#             unit = params.phi_theta.unit
#         )
#     )
#     lambda_theta = models.DimensionlessLuminosity.from_xm(
#         xm = xm,
#         tau_theta_array = params.tau_theta,
#         integral_table = integral_table
#     )
#     lnu = lambda_theta.to_lnu(
#         lnu_theta = params.lnu_theta
#     )
#     fnu_src = models.SourceframeSpecificFlux.from_lnu(
#         lnu = lnu,
#         distance = params.distance
#     )
#     fnu_obs = fnu_src.to_observerframe(
#         doppler_factor = params.doppler_delta
#     )
#     lc = CalculationLightcurves(
#         params = params,
#         nu = nu,
#         t = t,
#         fnu = fnu_obs
#     )
#     return lc.with_zero_boundary()
#
