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
from module.models import ChiSquaredParameters, ColumnNames, DefaultUnits, FluxName, PhaseParameter, ScenarioParameter, ThermalSynchrotronUtils, calculate_xm
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

# def ensure_string_keys(
#     mapping: Mapping[Hashable, Any],
# ) -> dict[str, Any]:
#     result: dict[str, Any] = {}
#
#     for key, value in mapping.items():
#         if not isinstance(key, str):
#             raise TypeError(
#                 f"辞書のキーはstrでなければなりません: {key!r}"
#             )
#
#         result[key] = value
#
#     return result
#
# @dataclass(frozen=True, slots=True)
# class UnitMetadata:
#     t_unit: str
#     fnu_unit: str
#
# @dataclass(frozen=True, slots=True)
# class ChevalierParameters:
#     a_wind: QuantityData
#     beta_sh: float
#     phi_peak: QuantityData
#     phi_theta: QuantityData
#     lnu_theta: QuantityData
#     tau_theta: float
#     doppler_delta: float
#     eps_b: float
#     eps_th: float
#     mu: float
#     mu_e: float
#     distance: QuantityData
#     chi2: float
#
#     @classmethod
#     def from_dict(
#         cls,
#         metadata:dict[str,Any]
#     )->Self:
#         a_wind = QuantityData(
#             value = metadata[ParameterNames.A_WIND],
#             unit = metadata[ParameterNames.A_WIND_UNIT]
#         )
#         phi_peak = QuantityData(
#             value = metadata[ParameterNames.PHI_PEAK],
#             unit = metadata[ParameterNames.PHI_UNIT]
#         )
#         phi_theta = QuantityData(
#             value = metadata[ParameterNames.PHI_THETA],
#             unit = metadata[ParameterNames.PHI_UNIT]
#         )
#         lnu_theta = QuantityData(
#             value = metadata[ParameterNames.LNU_THETA],
#             unit = metadata[ParameterNames.LNU_UNIT]
#         )
#         distance = QuantityData(
#             value = metadata[ParameterNames.DISTANCE_VALUE],
#             unit = metadata[ParameterNames.DISTANCE_UNIT]
#         )
#         return cls(
#             a_wind = a_wind,
#             beta_sh = metadata[ParameterNames.BETA_SH],
#             phi_peak=phi_peak,
#             phi_theta=phi_theta,
#             lnu_theta=lnu_theta,
#             tau_theta=metadata[ParameterNames.TAU_THETA],
#             doppler_delta=metadata[ParameterNames.DOPPLER_DELTA],
#             eps_b=metadata[ParameterNames.EPS_B],
#             eps_th=metadata[ParameterNames.EPS_TH],
#             mu=metadata[ParameterNames.MU],
#             mu_e=metadata[ParameterNames.MU_E],
#             chi2=metadata[ParameterNames.CHI2],
#             distance=distance
#         )
#
#     def to_dict(
#         self,
#         *,
#         a_wind_unit: str,
#         phi_unit:str,
#         lnu_unit:str,
#     )->dict[str,float|str]:
#         return {
#             ParameterNames.A_WIND: self.a_wind.value_in(a_wind_unit),
#             ParameterNames.A_WIND_UNIT: a_wind_unit,
#             ParameterNames.BETA_SH: self.beta_sh,
#             ParameterNames.PHI_PEAK: self.phi_peak.value_in(phi_unit),
#             ParameterNames.PHI_THETA: self.phi_theta.value_in(phi_unit),
#             ParameterNames.PHI_UNIT: phi_unit,
#             ParameterNames.LNU_THETA: self.lnu_theta.value_in(lnu_unit),
#             ParameterNames.LNU_UNIT: lnu_unit,
#             ParameterNames.TAU_THETA: self.tau_theta,
#             ParameterNames.DOPPLER_DELTA: self.doppler_delta,
#             ParameterNames.EPS_B: self.eps_b,
#             ParameterNames.EPS_TH: self.eps_th,
#             ParameterNames.MU: self.mu,
#             ParameterNames.MU_E: self.mu_e,
#             ParameterNames.DISTANCE_VALUE: self.distance.value,
#             ParameterNames.DISTANCE_UNIT: self.distance.unit,
#             ParameterNames.CHI2: self.chi2,
#         }
#
# @dataclass(frozen=True,slots=True)
# class CandidateModelRows:
#     model_index: NDArray[np.float64]
#     nu: QuantityData
#
# @dataclass(frozen=True,slots=True)
# class CandidateModelLightcurveConfig:
#     phi: models.PhaseParameter
#
# @dataclass(frozen=True,slots=True)
# class CandidateModelInput:
#     eps_b: float
#     eps_th: float
#     mu_e_mu: float
#     distance_value: float
#     distance_unit: str
#     phi_unit: str
#     lnu_unit: str
#     fnu_unit: str
#     a_wind_unit: str
#     nu_unit: str
#
#     @classmethod
#     def from_df_attrs(
#         cls,
#         df:pd.DataFrame
#     )->Self:
#         return cls(
#             **ensure_string_keys(df.attrs)
#         )
#
#     @property
#     def distance(self):
#         return QuantityData(
#             value = self.distance_value,
#             unit = self.distance_unit
#         )
#
# @dataclass(frozen=True,slots=True)
# class PhaseParameterGrid(QuantityGrid):
#
#     @classmethod
#     def build(
#         cls,     
#         t: QuantityArray1D,
#         nu: QuantityArray1D
#     )->Self:
#         return cls.from_1DArrays(
#             row=nu,
#             column=t
#         )
#
#     def to_xm(
#         self,
#         phi_th: QuantityArray1D
#     )->FloatGrid:
#         return self.decompose_by_column(phi_th)
#
#     def to_ln_xm(
#         self,
#         phi_th: QuantityArray1D
#     )->FloatGrid:
#         return self.decompose_by_column(phi_th).ln
#
#     def to_log10_xm(
#         self,
#         phi_th: QuantityArray1D
#     )->FloatGrid:
#         return self.decompose_by_column(phi_th).log10
#
# @dataclass(frozen=True,slots=True)
# class SpecificLuminosityGrid(QuantityGrid):
#     @classmethod
#     def build(
#         cls,
#         lambda_grid: FloatGrid,
#         lnu_th: QuantityArray1D
#     )->Self:
#         return cls.from_FloatGrid(
#             grid=lambda_grid,
#             row_quantity=lnu_th
#         )
#
# @dataclass(frozen=True,slots=True)
# class ObserverframeSpecificFluxGrid(QuantityGrid):
#     def add_bg(
#         self,
#         fnu_bg:QuantityArray1D
#     )->Self:
#         q = self.quantity + fnu_bg.quantity
#         return type(self).from_quantity(q)
#
# @dataclass(frozen=True,slots=True)
# class SourceframeSpecificFluxGrid(QuantityGrid):
#     @classmethod
#     def from_lnu(
#         cls,
#         lnu: QuantityGrid,
#         distance: QuantityData
#     )->Self:
#         fnu = quantity_converter.lnu_into_fnu(
#             lnu = lnu.quantity,
#             distance = distance.quantity
#         )
#         return cls.from_quantity(fnu)
#
#     def to_observerframe(
#         self,
#         doppler_factor:QuantityArray1D
#     )->ObserverframeSpecificFluxGrid:
#         return ObserverframeSpecificFluxGrid(
#             values = self.values * (doppler_factor.values[:,None])**3,
#             unit = self.unit
#         )
#
# @dataclass(frozen=True,slots=True)
# class CandidateModelParameters:
#     metadata: CandidateModelInput
#     model_index: IntVector
#     nu: QuantityArray1D
#     phi_th: QuantityArray1D
#     theta0: QuantityArray1D
#     theta:  QuantityArray1D
#     tau_th: QuantityArray1D
#     lnu_th: QuantityArray1D
#     a_wind: QuantityArray1D
#     beta_sh: QuantityArray1D
#     doppler_delta: QuantityArray1D
#
#     @classmethod
#     def from_df(
#         cls,
#         df:pd.DataFrame
#     )->Self:
#         metadata = CandidateModelInput.from_df_attrs(df)
#         phi_unit = metadata.phi_unit
#         fnu_unit = metadata.fnu_unit
#         lnu_unit = metadata.lnu_unit
#         nu_unit = metadata.nu_unit
#         a_wind_unit = metadata.a_wind_unit
#
#         def import_qa_from_the_df(
#             column: str,
#             unit: str
#         )->QuantityArray1D:
#             return QuantityArray1D.from_df_unit(df,column,unit=unit)
#
#         return cls(
#             metadata = metadata,
#             model_index = IntVector.from_df(
#                 df,
#                 index_name=models.ColumnNames.MODEL_INDEX
#             ),
#             nu = import_qa_from_the_df(
#                 column=models.ColumnNames.NU,
#                 unit=nu_unit
#             ),
#             phi_th = import_qa_from_the_df(
#                 column=models.ColumnNames.PHI_TH,
#                 unit=phi_unit
#             ),
#             theta0 = import_qa_from_the_df(
#                 column=models.ColumnNames.THETA0,
#                 unit=""
#             ),
#             theta = import_qa_from_the_df(
#                 column=models.ColumnNames.THETA,
#                 unit=""
#             ),
#             tau_th = import_qa_from_the_df(
#                 column=models.ColumnNames.TAU_THETA,
#                 unit=""
#             ),
#             lnu_th = import_qa_from_the_df(
#                 column=models.ColumnNames.LNU_THETA,
#                 unit=lnu_unit
#             ),
#             a_wind = import_qa_from_the_df(
#                 column=models.ColumnNames.A_WIND,
#                 unit=a_wind_unit
#             ),
#             beta_sh = import_qa_from_the_df(
#                 column=models.ColumnNames.BETA_SH,
#                 unit=""
#             ),
#             doppler_delta = import_qa_from_the_df(
#                 column=models.ColumnNames.DOPPLER_DELTA,
#                 unit=""
#             ),
#         )
#
# @dataclass(frozen=True,slots=True)
# class CandidateModelLightcurveGrid:
#     metadata: CandidateModelInput
#     model_index: IntVector
#     nu: QuantityArray1D
#     t: QuantityArray1D
#     lnu: SpecificLuminosityGrid
#     fnu_src_frame: SourceframeSpecificFluxGrid
#     fnu_obs_frame: ObserverframeSpecificFluxGrid
#
#     def to_df(self) -> pd.DataFrame:
#         metadata = self.metadata
#         model_index = self.model_index.values
#         t = self.t.quantity
#         nu = self.nu.to_value_in(metadata.nu_unit)
#         fnu = self.fnu_obs_frame.to_value_in(metadata.fnu_unit)
#         lnu = self.lnu.to_value_in(metadata.lnu_unit)
#         fnu_src_frame = self.fnu_src_frame.to_value_in(metadata.fnu_unit)
#
#         n_model = model_index.size
#         n_time = t.size
#
#         if fnu.shape != (n_model, n_time):
#             raise ValueError(
#                 "fnu shape must be "
#                 f"({n_model}, {n_time}), "
#                 f"but got {fnu.shape}"
#             )
#
#         candidate_model_index = np.arange(n_model)
#
#         df = pd.DataFrame({
#             models.ColumnNames.CANDIDATE_INDEX: np.repeat(
#                 candidate_model_index,
#                 n_time,
#             ),
#             models.ColumnNames.NU: np.repeat(
#                 nu,
#                 n_time,
#             ),
#             models.ColumnNames.MODEL_INDEX: np.repeat(
#                 model_index,
#                 n_time,
#             ),
#             models.ColumnNames.T: np.tile(
#                 t.value,
#                 n_model,
#             ),
#             models.ColumnNames.FNU_OBS_FRAME: fnu.ravel(),
#             models.ColumnNames.FNU_SRC_FRAME: fnu_src_frame.ravel(),
#             models.ColumnNames.LNU: lnu.ravel()
#         })
#         df.attrs = {
#             models.MetadataNames.T_UNIT: self.t.unit,
#             **asdict(metadata)
#         }
#
#         return df
#
# @dataclass(frozen=True,slots=True)
# class CandidateModelLightcurveMetadata(CandidateModelInput):
#     t_unit: str
#
# @dataclass(frozen=True,slots=True)
# class CandidateModelBinnedLightcurve:
#     metadata: CandidateModelLightcurveMetadata
#     candidate_model_index: int
#     model_index: int
#     nu: QuantityData
#     fnu_flare: QuantityArray1D
#     t_center: QuantityArray1D
#     t_left: QuantityArray1D
#     t_right: QuantityArray1D
#
#
@dataclass(frozen=True,slots=True)
class ChiSquaredResult:
    model_index: int
    nu: float
    chi_squared: float
    ndof: int

    def to_series(
        self
    )->pd.Series:
        return pd.Series(asdict(self))

    @classmethod
    def calculate(
        cls,
        model_index: int,
        nu_value: float,
        fnu_model: QuantityArray1D,
        fnu_data: QuantityArray1D,
        fnu_err: QuantityArray1D,
        *,
        n_params: int,
        fnu_unit: str = DefaultUnits.fnu,
    )->Self:
        n_data = fnu_model.length

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
            model = fnu_model.to_floatarray1d(fnu_unit),
            observed = fnu_data.to_floatarray1d(fnu_unit),
            error = fnu_err.to_floatarray1d(fnu_unit)
        )

        return cls(
            model_index=model_index,
            nu=nu_value,
            chi_squared=chi2,
            ndof=n_data-n_params
        )

@dataclass(frozen=True,slots=True)
class BinnedLightcurve:
    model_index: int
    nu: QuantityData
    t_center: QuantityArray1D
    t_left: QuantityArray1D
    t_right: QuantityArray1D
    fnu_no_bg: QuantityArray1D

    def fnu_with_bg(
        self,
        fnu_bg: QuantityArray1D
    )->QuantityArray1D:
        return QuantityArray1D.from_quantity(
            self.fnu_no_bg.quantity
            + fnu_bg.quantity
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
        return QuantityArray1D(
            values=bin_edges,
            unit=t_unit
        )

    def to_df(
        self,
        fnu_bg: QuantityArray1D,
        *,
        nu_unit: str = DefaultUnits.nu,
        t_unit: str = DefaultUnits.t,
        fnu_unit: str = DefaultUnits.fnu,
    )->pd.DataFrame:
        df = pd.DataFrame({
            models.ColumnNames.MODEL_INDEX: self.model_index,
            models.ColumnNames.NU: self.nu.value_in(nu_unit),
            models.ColumnNames.T_CENTER: self.t_center.to_value_in(t_unit),
            models.ColumnNames.T_RIGHT: self.t_right.to_value_in(t_unit),
            models.ColumnNames.T_LEFT: self.t_left.to_value_in(t_unit),
            models.FluxName.FNU_WITH_BG: self.fnu_with_bg(fnu_bg).to_value_in(fnu_unit),
        })
        df.attrs[MetadataNames.NU_UNIT] = nu_unit
        df.attrs[MetadataNames.FNU_UNIT] = fnu_unit
        df.attrs[MetadataNames.T_UNIT] = t_unit
        return df

@dataclass(frozen=True,slots=True)
class ContinuousLightcurve:
    model_index: int
    nu: QuantityData
    t: QuantityArray1D
    fnu_net: QuantityArray1D

    def insert_zero_boundary(
        self
    )->Self:
        """
            配列の左端に時刻ゼロでの境界条件を挿入する
        """
        t_values = self.t.values
        fnu_values = self.fnu_net.values

        if t_values[0] == 0.0:
            return self

        if t_values[0] < 0.0:
            raise ValueError(
                "計算光度曲線に負の時刻が含まれています."
                f"t_min={t_values[0]}"
            )

        return type(self)(
            model_index=self.model_index,
            t=QuantityArray1D(
                values=np.concatenate((
                    np.array([0.0], dtype=np.float64),
                    t_values,
                )),
                unit=self.t.unit,
            ),
            fnu_net=QuantityArray1D(
                values=np.concatenate((
                    np.array([0.0], dtype=np.float64),
                    fnu_values,
                )),
                unit=self.fnu_net.unit,
            ),
            nu=self.nu,
        )

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

    def bin_average(
        self,
        bin_config: observation.Binning,
        drop_incomplete_bin: bool = True
    )->BinnedLightcurve:

        t_unit = bin_config.t_unit
        fnu_unit = self.fnu_net.unit

        ## 計算光度曲線が観測の時間範囲を覆っているかcheck
        t = self.t
        fnu = self.fnu_net
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

        return BinnedLightcurve(
            model_index= self.model_index,
            nu = self.nu,
            t_center=QuantityArray1D(t_center,t_unit),
            t_left=QuantityArray1D(t_left,t_unit),
            t_right=QuantityArray1D(t_right,t_unit),
            fnu_no_bg=fnu_average,
        )

    @classmethod
    def from_df(
        cls,
        model_index: int,
        nu: QuantityData,
        df: pd.DataFrame
    )-> Self:
        fnu_unit = df.attrs[MetadataNames.FNU_UNIT]
        t_unit = df.attrs[MetadataNames.T_UNIT]
        return cls(
            model_index=model_index,
            nu=nu,
            fnu_net=QuantityArray1D(
                df[models.ColumnNames.FNU_TOTAL]
                    .to_numpy(dtype=np.float64),
                unit=fnu_unit,
            ),
            t=QuantityArray1D(
                df[models.ColumnNames.T]
                    .to_numpy(dtype=np.float64),
                unit=t_unit,
            ),
        )

# @dataclass(frozen=True,slots=True)
# class CandidateModelLightcurveContinuous:
#     metadata: CandidateModelLightcurveMetadata
#     candidate_model_index: int
#     model_index: int
#     nu: QuantityData
#     t: QuantityArray1D
#     fnu_obs_frame: QuantityArray1D
#
#     @classmethod
#     def from_df(
#         cls,
#         candidate_model_index: int,
#         df:pd.DataFrame
#     )->Self:
#         metadata = CandidateModelLightcurveMetadata.from_df_attrs(df)
#         nu_unit = metadata.nu_unit
#         nu_array = QuantityArray1D.from_df(
#             df,models.ColumnNames.NU,
#             unit_key=models.MetadataNames.NU_UNIT
#         )
#         model_index = IntVector.from_df(
#             df,models.ColumnNames.MODEL_INDEX
#         ).as_scalar()
#
#         return cls(
#             metadata = metadata,
#             candidate_model_index=candidate_model_index,
#             model_index = model_index,
#             t = QuantityArray1D.from_df_unit(
#                 df,models.ColumnNames.T,metadata.t_unit
#             ),
#             nu = nu_array.index(0),
#             fnu_obs_frame = QuantityArray1D.from_df_unit(
#                 df,models.ColumnNames.FNU_OBS_FRAME,metadata.fnu_unit
#             ),
#         )
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
#         t_model: QuantityArray1D,
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
#     def insert_zero_boundary(
#         self
#     )->Self:
#         """
#             配列の左端に時刻ゼロでの境界条件を挿入する
#         """
#         t_values = self.t.values
#         fnu_values = self.fnu_obs_frame.values
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
#         return type(self)(
#             metadata=self.metadata,
#             candidate_model_index=self.candidate_model_index,
#             model_index = self.model_index,
#             t=QuantityArray1D(
#                 values=np.concatenate((
#                     np.array([0.0], dtype=np.float64),
#                     t_values,
#                 )),
#                 unit=self.t.unit,
#             ),
#             fnu_obs_frame=QuantityArray1D(
#                 values=np.concatenate((
#                     np.array([0.0], dtype=np.float64),
#                     fnu_values,
#                 )),
#                 unit=self.fnu_obs_frame.unit,
#             ),
#             nu=self.nu,
#         )
#
#     def bin_average(
#         self,
#         bin_config: observation.Binning,
#         drop_incomplete_bin: bool = True
#     )->CandidateModelBinnedLightcurve:
#
#         t_unit = bin_config.t_unit
#         fnu_unit = self.fnu_obs_frame.unit
#
#         ## 計算光度曲線が観測の時間範囲を覆っているかcheck
#         t = self.t
#         fnu = self.fnu_obs_frame
#         self.validate_time_coverage(t,bin_config)
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
#         ) -> FloatArray:
#             return np.interp(
#                 t_evaluate,
#                 t_model,
#                 fnu_model,
#             )
#
#         # shapeはいずれも(M, B)
#         fnu_left = interpolate_flux(t_left)
#         fnu_right = interpolate_flux(t_right)
#
#         # shapeはいずれも(M, B)
#         fnu_left = interpolate_flux(t_left)
#         fnu_right = interpolate_flux(t_right)
#
#         fnu_average_by_bin: list[float] = []
#
#         for i, (left, right) in enumerate(
#             zip(t_left, t_right, strict=True)
#         ):
#             inside = (
#                 (t_model > left)
#                 & (t_model < right)
#             )
#
#             t_bin = np.concatenate((
#                 np.array([left], dtype=np.float64),
#                 t_model[inside],
#                 np.array([right], dtype=np.float64),
#             ))
#
#             fnu_bin = np.concatenate((
#                 np.array([fnu_left[i]], dtype=np.float64),
#                 fnu_model[inside],
#                 np.array([fnu_right[i]], dtype=np.float64),
#             ))
#
#             flux_integral = np.trapezoid(
#                 fnu_bin,
#                 x=t_bin,
#             )
#
#             fnu_average_by_bin.append(
#                 float(flux_integral / (right - left))
#             )
#
#         fnu_average = QuantityArray1D(
#             values=np.asarray(
#                 fnu_average_by_bin,
#                 dtype=np.float64,
#             ),
#             unit=fnu_unit,
#         )
#
#         return CandidateModelBinnedLightcurve(
#             metadata=self.metadata,
#             candidate_model_index=self.candidate_model_index,
#             model_index= self.model_index,
#             nu = self.nu,
#             t_center=QuantityArray1D(t_center,t_unit),
#             t_left=QuantityArray1D(t_left,t_unit),
#             t_right=QuantityArray1D(t_right,t_unit),
#             fnu_flare=fnu_average,
#         )
#
#     def fnu_with_bg(
#         self,
#         fnu_bg:QuantityData
#     )->QuantityArray1D:
#         fnu = self.fnu_obs_frame.quantity + fnu_bg.quantity
#         return QuantityArray1D.from_quantity(fnu)
#
#     def plot(
#         self,
#         plotter: plot_utils.Plotter,
#         style: plot_utils.LineStyle,
#         *,
#         t_unit: str,
#         fnu_unit: str,
#         offset: float
#     ):
#         x = self.t.to_value_in(t_unit)
#         y = self.fnu_obs_frame.to_value_in(fnu_unit) + offset
#         return plotter.line(
#             x,y,style
#         )
#
#
# @dataclass(frozen=True, slots=True)
# class CandidateModelTotalLightcurve:
#     metadata: CandidateModelLightcurveMetadata
#     candidate_model_index: int
#     model_index: int
#     nu: QuantityData
#     t_center: QuantityArray1D
#     t_left: QuantityArray1D
#     t_right: QuantityArray1D
#     fnu_flare: QuantityArray1D
#     fnu_bg: QuantityArray1D
#     fnu_total: QuantityArray1D
#
#     @classmethod
#     def from_df(
#         cls,
#         candidate_model_index:int,
#         metadata:CandidateModelLightcurveMetadata,
#         df:pd.DataFrame
#     ):
#         nu_arr = QuantityArray1D.from_df(
#             df,models.ColumnNames.NU,metadata.nu_unit
#         )
#         model_index = IntVector.from_df(
#             df,models.ColumnNames.MODEL_INDEX
#         ).as_scalar()
#         return cls(
#             metadata = metadata,
#             candidate_model_index=candidate_model_index,
#             model_index = model_index,
#             t_center = QuantityArray1D.from_df(
#                 df,models.ColumnNames.T_CENTER,metadata.t_unit
#             ),
#             t_left = QuantityArray1D.from_df(
#                 df,models.ColumnNames.T_LEFT,metadata.t_unit
#             ),
#             t_right = QuantityArray1D.from_df(
#                 df,models.ColumnNames.T_RIGHT,metadata.t_unit
#             ),
#             nu = nu_arr.index(0),
#             fnu_flare = QuantityArray1D.from_df(
#                 df,models.ColumnNames.FNU_FLARE,metadata.fnu_unit
#             ),
#             fnu_bg = QuantityArray1D.from_df(
#                 df,models.ColumnNames.FNU_BG,metadata.fnu_unit
#             ),
#             fnu_total = QuantityArray1D.from_df(
#                 df,models.ColumnNames.FNU_TOTAL,metadata.fnu_unit
#             ),
#         )
#
#     @classmethod
#     def from_binned_lightcurve(
#         cls,
#         lightcurve: CandidateModelBinnedLightcurve,
#         fnu_bg: QuantityArray1D,
#     ) -> Self:
#         if lightcurve.t_center.length != fnu_bg.length:
#             raise ValueError(
#                 "光度曲線とbackgroundの配列長が"
#                 "一致していません."
#             )
#
#         fnu_total = QuantityArray1D.from_quantity(
#             lightcurve.fnu_flare.quantity
#             + fnu_bg.quantity
#         )
#
#         return cls(
#             metadata=lightcurve.metadata,
#             candidate_model_index=(
#                 lightcurve.candidate_model_index
#             ),
#             model_index = lightcurve.model_index,
#             nu=lightcurve.nu,
#             t_center=lightcurve.t_center,
#             t_left=lightcurve.t_left,
#             t_right=lightcurve.t_right,
#             fnu_flare=lightcurve.fnu_flare,
#             fnu_bg=fnu_bg,
#             fnu_total=fnu_total,
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
#     def plot(
#         self,
#         plotter: plot_utils.Plotter,
#         style: LineStyle,
#         *,
#         t_unit: str,
#         fnu_unit: str
#     )->StepPatch:
#         bin_edges_q = self.bin_edges
#         return plotter.step(
#             values=self.fnu_total.to_value_in(fnu_unit),
#             edges=bin_edges_q.to_value_in(t_unit),
#             style=style,
#             baseline=self.fnu_bg.to_value_in(fnu_unit)[0]
#         )
#
# @dataclass(frozen=True, slots=True)
# class CandidateModelFitResult:
#     lightcurve: CandidateModelTotalLightcurve
#     chi2: float
#     ndof: int
#
#     @classmethod
#     def from_binned_lightcurve(
#         cls,
#         lightcurve: CandidateModelTotalLightcurve,
#         fnu_data: QuantityArray1D,
#         fnu_err: QuantityArray1D,
#         *,
#         n_params: int
#     )->Self:
#         n_data = lightcurve.fnu_total.length
#         fnu_unit = lightcurve.metadata.fnu_unit
#
#         if not (
#             fnu_data.length
#             == fnu_err.length
#             == n_data
#         ):
#             raise ValueError(
#             "モデル, 観測値, 観測誤差の"
#             "配列長が一致していません."
#         )
#         chi2 = CalculateChiSquared.from_vector(
#             model = lightcurve.fnu_total.to_floatarray1d(fnu_unit),
#             observed = fnu_data.to_floatarray1d(fnu_unit),
#             error = fnu_err.to_floatarray1d(fnu_unit)
#         )
#
#         return cls(
#             lightcurve=lightcurve,
#             chi2=chi2,
#             ndof=n_data-n_params
#         )
#
#     def to_df(
#         self,
#     )->pd.DataFrame:
#         metadata = self.lightcurve.metadata
#         n_bin = self.lightcurve.t_center.length
#
#         t_unit = metadata.t_unit
#         nu_unit = metadata.nu_unit
#         fnu_unit = metadata.fnu_unit
#
#         if not (
#             self.lightcurve.t_left.length
#             == self.lightcurve.t_right.length
#             == self.lightcurve.fnu_total.length
#             == n_bin
#         ):
#             raise ValueError(
#                 "bin平均光度曲線の配列長が一致しません."
#             )
#
#         df = pd.DataFrame({
#             models.ColumnNames.CANDIDATE_INDEX: np.full(
#                 n_bin,
#                 self.lightcurve.candidate_model_index,
#                 dtype=np.int64
#             ),
#             models.ColumnNames.MODEL_INDEX: np.full(
#                 n_bin,
#                 self.lightcurve.model_index,
#                 dtype=np.int64
#             ),
#             models.ColumnNames.NU: np.full(
#                 n_bin,
#                 self.lightcurve.nu.value_in(nu_unit),
#                 dtype=np.float64
#             ),
#             models.ColumnNames.T_CENTER: self.lightcurve.t_center.to_value_in(t_unit),
#             models.ColumnNames.T_LEFT: self.lightcurve.t_left.to_value_in(t_unit),
#             models.ColumnNames.T_RIGHT: self.lightcurve.t_right.to_value_in(t_unit),
#             models.ColumnNames.FNU_FLARE: self.lightcurve.fnu_flare.to_value_in(fnu_unit),
#             models.ColumnNames.FNU_BG: self.lightcurve.fnu_bg.to_value_in(fnu_unit),
#             models.ColumnNames.FNU_TOTAL: self.lightcurve.fnu_total.to_value_in(fnu_unit),
#             models.ChiSquaredParameters.CHI_SQUARED: self.chi2,
#             models.ChiSquaredParameters.N_DOF: self.ndof,
#         })
#         df.attrs.update({**asdict(metadata)})
#         return df
#
# @dataclass(frozen=True, slots=True)
# class CandidateModelParameter(CandidateModelInput):
#     a_wind_value: float
#     beta_sh: float
#
# @dataclass(frozen=True, slots=True)
# class CandidateModelLightcurve:
#     nu: QuantityData
#     df: pd.DataFrame
#
#     @property
#     def t_left(self):
#         return QuantityArray1D.from_df(
#             df=self.df,
#             column=models.ColumnNames.T_LEFT,
#             unit_key=models.MetadataNames.T_UNIT
#         )
#     @property
#     def t_right(self):
#         return QuantityArray1D.from_df(
#             df=self.df,
#             column=models.ColumnNames.T_RIGHT,
#             unit_key=models.MetadataNames.T_UNIT
#         )
#
#     @property
#     def model_index(self):
#         idx = self.df.attrs[models.ColumnNames.MODEL_INDEX]
#         return idx
#
#     @property
#     def candidate_model_index(self):
#         idx = self.df.attrs[models.ColumnNames.CANDIDATE_INDEX]
#         return idx
#
#     @property
#     def bin_edges(
#         self,
#     )->QuantityArray1D:
#         t_unit = self.t_left.unit
#         t_left = self.t_left.to_value_in(t_unit)
#         t_right = self.t_right.to_value_in(t_unit)
#         bin_edges = np.concatenate(
#             (
#                 t_left,
#                 np.array([t_right[-1]]),
#             )
#         )
#         return QuantityArray1D(bin_edges,unit=t_unit)
#
#     @property
#     def fnu_total(self):
#         return QuantityArray1D.from_df(
#             df=self.df,
#             column=models.ColumnNames.FNU_TOTAL,
#             unit_key=models.MetadataNames.FNU_UNIT
#         )
#
#     @property
#     def fnu_bg(self):
#         return QuantityData.from_attrs(
#             df=self.df,
#             key=models.ColumnNames.FNU_BG,
#             unit_key=models.MetadataNames.FNU_UNIT
#         )
#
# @dataclass(frozen=True, slots=True)
# class CandidateModelLightcurveFrequencyGroup:
#     nu: QuantityData
#     models: dict[Hashable,pd.DataFrame]
#
#     def select_model(
#         self,
#         model_index:Hashable
#     )->CandidateModelLightcurve:
#         df=self.models[model_index]
#         attrs = df.attrs
#         if not isinstance(model_index,int):
#             raise TypeError(
#                 f"model_indexをint型へ変換できません:{model_index}"
#             )
#         attrs[models.ColumnNames.MODEL_INDEX] = model_index
#         return CandidateModelLightcurve(
#             nu=self.nu,
#             df=df,
#         )
#
#     def select_unique(
#         self,
#     ) -> CandidateModelLightcurve:
#         if len(self.models) != 1:
#             raise ValueError(
#                 "models must contain exactly one model, "
#                 f"but got {len(self.models)}"
#             )
#
#         model_index = next(iter(self.models))
#
#         return self.select_model(model_index)
#
#     def devide_into_models(
#         self,
#     )->list[CandidateModelLightcurve]:
#         groups:list[CandidateModelLightcurve] = []
#         dfs_model = {}
#         for model_index,df in self.models.items():
#             model = self.select_model(model_index)
#             groups.append(model)
#
#         return groups
#
#     def index_minimum_chi2(self) -> Hashable:
#         chi_squared:dict[Hashable,float] = {
#             key: float(df.attrs[models.ChiSquaredParameters.CHI_SQUARED])
#             for key, df in self.models.items()
#         }
#
#         best_key = min(
#             chi_squared,
#             key=lambda key:chi_squared[key],
#         )
#
#         return best_key
#
#     def to_df_minimum_chi2(self)->pd.DataFrame:
#         index_minimum_chi2 = self.index_minimum_chi2()
#         df = self.models[index_minimum_chi2]
#         metadata_columns = [
#             models.ColumnNames.FNU_BG,
#             models.ChiSquaredParameters.CHI_SQUARED,
#             models.ChiSquaredParameters.N_DOF,
#             models.ColumnNames.CANDIDATE_INDEX,
#         ]
#
#         for _, df in self.models.items():
#             df = df.copy()
#
#             for column in metadata_columns:
#                 value = df.attrs.pop(column)
#                 df[column] = value
#         df[models.ColumnNames.NU] = self.nu.value
#         df[models.ColumnNames.MODEL_INDEX] = index_minimum_chi2
#
#         return df
#
# @dataclass(frozen=True, slots=True)
# class CandidateModelLightcurveTable:
#     metadata: CandidateModelLightcurveMetadata
#     frequency: dict[Hashable,pd.DataFrame]
#
#     @classmethod
#     def from_df_long(
#         cls,
#         df_long: pd.DataFrame
#     ) -> Self:
#         metadata = CandidateModelLightcurveMetadata.from_df_attrs(df_long)
#
#         nu_key = models.ColumnNames.NU
#         dfs_nu = {
#             nu: df.drop(columns=nu_key)
#             for nu,df in df_long.groupby(nu_key)
#         }
#         return cls(
#             metadata=metadata,
#             frequency=dfs_nu
#         )
#
#     def devide_into_frequency_groups(
#         self,
#     )->list[CandidateModelLightcurveFrequencyGroup]:
#         groups:list[CandidateModelLightcurveFrequencyGroup] = []
#         for nu_value in self.frequency.keys():
#             df_nu = self.frequency[nu_value]
#
#             model_metadata_columns = [
#                 models.ColumnNames.FNU_BG,
#                 models.ChiSquaredParameters.CHI_SQUARED,
#                 models.ChiSquaredParameters.N_DOF,
#                 models.ColumnNames.CANDIDATE_INDEX,
#             ]
#             dfs_model = {}
#             for idx,df in df_nu.groupby(
#                 models.ColumnNames.MODEL_INDEX
#             ):
#                 df = df.drop(
#                     columns=models.ColumnNames.MODEL_INDEX
#                 )
#
#                 attrs = {}
#
#                 for column in model_metadata_columns:
#                     values = df[column].unique()
#
#                     if len(values) != 1:
#                         raise ValueError(
#                             f"{column} must have exactly one unique value, "
#                             f"but got {values}"
#                         )
#
#                     attrs[column] = values[0]
#
#                 df = df.drop(columns=model_metadata_columns)
#                 df.attrs.update(attrs)
#
#                 dfs_model[idx] = df
#
#             if not isinstance(nu_value,float):
#                 raise TypeError(
#                     f"nuをfloat型へ変換できません:{nu_value}"
#                 )
#
#             groups.append(
#                 CandidateModelLightcurveFrequencyGroup(
#                     nu=QuantityData(
#                         value=nu_value,
#                         unit=self.metadata.nu_unit
#                     ),
#                     models = dfs_model
#                 )
#             )
#
#         return groups
#
#
#
# @dataclass(frozen=True, slots=True)
# class CandidateModelMetadata(models.ScenarioParameter):
#     a_wind: QuantityData
#     beta_sh: QuantityData
#     nu: QuantityData
#
#     @classmethod
#     def from_df(
#         cls,
#         df_models:pd.DataFrame,
#         model_index:int
#     ):
#         row:pd.Series = df_models.loc[
#             df_models[
#                 models.ColumnNames.MODEL_INDEX
#             ] == model_index
#         ].iloc[0]
#         return cls(
#             eps_b=QuantityData.from_attrs(
#                 df_models,
#                 key=models.MetadataNames.EPS_B,
#             ),
#             eps_th=QuantityData.from_attrs(
#                 df_models,
#                 key=models.MetadataNames.EPS_TH,
#             ),
#             mu_e_mu=QuantityData.from_attrs(
#                 df_models,
#                 key=models.MetadataNames.MU_E_MU,
#             ),
#             distance=QuantityData.from_attrs(
#                 df_models,
#                 key=models.MetadataNames.DISTANCE_VALUE,
#                 unit_key=models.MetadataNames.DISTANCE_UNIT
#             ),
#             a_wind = QuantityData.from_series(
#                 row=row,
#                 column=models.ColumnNames.A_WIND,
#                 unit=df_models.attrs[ParameterNames.A_WIND_UNIT]
#             ),
#             beta_sh = QuantityData.from_series(
#                 row=row,
#                 column=models.ColumnNames.BETA_SH,
#                 unit=""
#             ),
#             nu = QuantityData.from_series(
#                 row=row,
#                 column=models.ColumnNames.NU,
#                 unit=df_models.attrs[models.MetadataNames.NU_UNIT]
#             ),
#         )
#
# @dataclass(frozen=True, slots=True)
# class CandidateModelLightcurveContinuousPlot:
#     candidate_model_index: int
#     model_index: int
#     nu: QuantityData
#     t: QuantityArray1D
#     fnu_obs_frame: QuantityArray1D
#
#     @classmethod
#     def from_df(
#         cls,
#         df:pd.DataFrame,
#         candidate_model_index: int
#     ):
#         df_selected:pd.DataFrame = df.loc[
#             df[models.ColumnNames.CANDIDATE_INDEX] == candidate_model_index
#         ]
#         return cls(
#             candidate_model_index=candidate_model_index,
#             model_index=
#                 IntVector.from_df(df_selected,models.ColumnNames.MODEL_INDEX).as_scalar(),
#             nu = QuantityArray1D.from_df(
#                 df_selected,
#                 models.ColumnNames.NU,
#                 unit_key=models.MetadataNames.NU_UNIT
#             ).index(0),
#             t = QuantityArray1D.from_df(
#                 df_selected,
#                 models.ColumnNames.T,
#                 unit_key=models.MetadataNames.T_UNIT
#             ),
#             fnu_obs_frame = QuantityArray1D.from_df(
#                 df_selected,
#                 models.ColumnNames.FNU_OBS_FRAME,
#                 unit_key=models.MetadataNames.FNU_UNIT
#             ),
#         )
#
#     def fnu_with_bg(
#         self,
#         fnu_bg:QuantityData
#     )->QuantityArray1D:
#         fnu = self.fnu_obs_frame.quantity + fnu_bg.quantity
#         return QuantityArray1D.from_quantity(fnu)
