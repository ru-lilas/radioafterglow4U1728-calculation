from collections.abc import Mapping
from pathlib import Path
from dataclasses import dataclass
from typing import Self, cast, Any, Hashable

from matplotlib.patches import StepPatch
from module import plot_utils
from module import models,dataframe_utils
from module.models import SynchrotronPhase, ThermalSynchrotronUtils, calculate_xm
from module.mydataclasses import QuantityArray, QuantityData
from module.mystatistics import CalculateChiSquared
from module.plot_utils import LineStyle
from module.types import FloatArray, FloatArray1D, FloatGrid
import astropy.units as u
import numpy as np
import pandas as pd
from module.utils import DataFrameUtils, Integrator,FileReader
from module import dataframe_processors, observation
from enum import StrEnum, auto
from matplotlib.axes import Axes
from cattrs import Converter
from module import inputs_as_dataclass

DATACLASS_CONVERTER = Converter()

class Columns(StrEnum):
    T_OBSERVER_FRAME = auto()
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

@dataclass(frozen=True,slots=True)
class InputValues:
    idx:int
    a_wind: float
    beta_sh:float
    eps_b:float
    eps_th:float
    mu:float
    mu_e:float
    distance: float
    theta: float
    phi_theta: float
    lnu_theta: float
    tau_theta: float
    doppler_delta: float

@dataclass(frozen=True,slots=True)
class InputUnits:
    a_wind: str
    phi_theta: str
    lnu_theta: str
    distance: str

    @classmethod
    def from_keyvalue(
            cls,
            path:Path
    )->Self:
        dict_data = FileReader.keyvalue(path)
        return DATACLASS_CONVERTER.structure(
            dict_data,
            cls
        )

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

@dataclass(frozen=True,slots=True)
class Input:
    values: InputValues
    units: InputUnits

    @classmethod
    def from_idx(
        cls,
        df:pd.DataFrame,
        idx: int
    ):
        row:pd.Series = DataFrameUtils.extract_row_by_index(df,idx)
        values_dict = ensure_string_keys(
            row.to_dict()
        )
        units_dict = ensure_string_keys(
            df.attrs
        )
        return cls(
            values = InputValues(
                **values_dict,
                idx=idx
            ),
            units = InputUnits(**units_dict)
        )

    @property
    def a_wind(self)->u.Quantity:
        return u.Quantity(self.values.a_wind,self.units.a_wind)

    @property
    def phi_theta(self)->u.Quantity:
        return u.Quantity(self.values.phi_theta,self.units.phi_theta)

    @property
    def lnu_theta(self)->u.Quantity:
        return u.Quantity(self.values.lnu_theta,self.units.lnu_theta)

    @property
    def distance(self)->u.Quantity:
        return u.Quantity(self.values.distance,self.units.distance)

    @property
    def tau_theta(self)->float:
        return self.values.tau_theta

    @property
    def doppler_delta(self)->float:
        return self.values.doppler_delta

@dataclass(frozen=True, slots=True)
class UnitMetadata:
    t_unit: str
    fnu_unit: str

@dataclass(frozen=True, slots=True)
class BestfitParameters:
    a_wind: QuantityData
    beta_sh: float
    phi_peak: QuantityData
    fnu_peak: QuantityData
    phi_theta: QuantityData
    lnu_theta: QuantityData
    tau_theta: float
    doppler_delta: float
    eps_b: float
    eps_th: float
    mu: float
    mu_e: float
    distance: QuantityData

@dataclass(frozen=True,slots=True)
class EstimatedParameters:
    a_wind: QuantityArray
    beta_sh: FloatArray
    phi_peak: QuantityArray
    fnu_peak: QuantityArray
    phi_theta: QuantityArray
    lnu_theta: QuantityArray
    tau_theta: FloatArray
    doppler_delta: FloatArray
    eps_b: float
    eps_th: float
    mu: float
    mu_e: float
    distance: QuantityData

    def __post_init__(self) -> None:
        arrays = {
            ParameterNames.A_WIND: self.a_wind.values,
            ParameterNames.BETA_SH: self.beta_sh,
            ParameterNames.PHI_PEAK: self.phi_peak.values,
            ParameterNames.FNU_PEAK: self.fnu_peak.values,
            ParameterNames.PHI_THETA: self.phi_theta.values,
            ParameterNames.LNU_THETA: self.lnu_theta.values,
            ParameterNames.TAU_THETA: self.tau_theta,
        }

        for name, array in arrays.items():
            if array.ndim != 1:
                raise ValueError(
                    f"{name}は1次元でなければなりません. "
                    f"shape={array.shape}"
                )

        sizes = {
            name: array.size
            for name, array in arrays.items()
        }

        if len(set(sizes.values())) != 1:
            raise ValueError(
                "推定パラメータの配列長が一致していません. "
                f"{sizes}"
            )

    @classmethod
    def from_csv(
        cls,
        path: Path
    )->Self:
        df = FileReader.table_from_csv(path)
        metadata = df.attrs
        return cls(
            a_wind = QuantityArray(
                values = dataframe_utils.extract_column_as_ndarray(
                    df,
                    ParameterNames.A_WIND
                ),
                unit = metadata[ParameterNames.A_WIND_UNIT],
            ),
            beta_sh = dataframe_utils.extract_column_as_ndarray(
                df,
                ParameterNames.BETA_SH
            ),
            phi_peak = QuantityArray(
                values = dataframe_utils.extract_column_as_ndarray(
                    df,
                    ParameterNames.PHI_PEAK
                ),
                unit = metadata[ParameterNames.PHI_UNIT]
            ),
            phi_theta = QuantityArray(
                values = dataframe_utils.extract_column_as_ndarray(
                    df,
                    ParameterNames.PHI_THETA
                ),
                unit = metadata[ParameterNames.PHI_UNIT]
            ),
            lnu_theta = QuantityArray(
                values = dataframe_utils.extract_column_as_ndarray(
                    df,
                    ParameterNames.LNU_THETA
                ),
                unit = metadata[ParameterNames.LNU_UNIT]
            ),
            tau_theta = dataframe_utils.extract_column_as_ndarray(
                    df,
                    ParameterNames.TAU_THETA
            ),
            doppler_delta = dataframe_utils.extract_column_as_ndarray(
                df,
                ParameterNames.DOPPLER_DELTA
            ),
            fnu_peak = QuantityArray(
                values = dataframe_utils.extract_column_as_ndarray(
                    df,
                    ParameterNames.FNU_PEAK
                ),
                unit = metadata[ParameterNames.FNU_UNIT]
            ),
            eps_b = metadata[ParameterNames.EPS_B],
            eps_th = metadata[ParameterNames.EPS_TH],
            mu = metadata[ParameterNames.MU],
            mu_e = metadata[ParameterNames.MU_E],
            distance = QuantityData(
                value = metadata[ParameterNames.DISTANCE_VALUE],
                unit = metadata[ParameterNames.DISTANCE_UNIT],
            )
        )

    def to_df(
        self,
        phi_unit,
        fnu_unit,
        lnu_unit
    )->pd.DataFrame:
        a_wind = self.a_wind
        df = pd.DataFrame({
            ParameterNames.A_WIND: a_wind.values,
            ParameterNames.BETA_SH: self.beta_sh,
            ParameterNames.PHI_PEAK: self.phi_peak.to_value_in(phi_unit),
            ParameterNames.FNU_PEAK: self.fnu_peak.to_value_in(fnu_unit),
            ParameterNames.PHI_THETA: self.phi_theta.to_value_in(phi_unit),
            ParameterNames.LNU_THETA: self.lnu_theta.to_value_in(lnu_unit),
            ParameterNames.TAU_THETA: self.tau_theta,
            ParameterNames.DOPPLER_DELTA: self.doppler_delta,
        })
        df.attrs[ParameterNames.EPS_B] = self.eps_b
        df.attrs[ParameterNames.EPS_TH] = self.eps_th
        df.attrs[ParameterNames.MU] = self.mu
        df.attrs[ParameterNames.MU_E] = self.mu_e
        df.attrs[ParameterNames.DISTANCE_VALUE] = self.distance.value
        df.attrs[ParameterNames.DISTANCE_UNIT] = self.distance.unit
        df.attrs[ParameterNames.PHI_UNIT] = phi_unit
        df.attrs[ParameterNames.FNU_UNIT] = fnu_unit
        df.attrs[ParameterNames.LNU_UNIT] = lnu_unit
        df.attrs[ParameterNames.A_WIND_UNIT] = a_wind.unit
        return df

    def extract_bestfit(
        self,
        idx: int
    )->BestfitParameters:
        a_wind_arr = self.a_wind
        phi_peak_arr = self.phi_peak
        phi_theta_arr = self.phi_theta
        lnu_theta_arr = self.lnu_theta
        fnu_peak_arr = self.fnu_peak
        a_wind = QuantityData(
            value = a_wind_arr.values[idx],
            unit = a_wind_arr.unit
        )
        phi_peak = QuantityData(
            value = phi_peak_arr.values[idx],
            unit = phi_peak_arr.unit
        )
        phi_theta = QuantityData(
            value = phi_theta_arr.values[idx],
            unit = phi_theta_arr.unit
        )
        lnu_theta = QuantityData(
            value = lnu_theta_arr.values[idx],
            unit = lnu_theta_arr.unit
        )
        fnu_peak = QuantityData(
            value = fnu_peak_arr.values[idx],
            unit = fnu_peak_arr.unit
        )
        return BestfitParameters(
            a_wind = a_wind,
            beta_sh = self.beta_sh[idx],
            phi_peak = phi_peak,
            fnu_peak = fnu_peak,
            phi_theta = phi_theta,
            lnu_theta = lnu_theta,
            tau_theta = self.tau_theta[idx],
            doppler_delta= self.doppler_delta[idx],
            eps_b=self.eps_b,
            eps_th = self.eps_th,
            mu = self.mu,
            mu_e = self.mu_e,
            distance=self.distance
        )

@dataclass(frozen=True, slots=True)
class BinnedBestfitLightcurve:
    params: BestfitParameters
    t_center:   QuantityArray
    t_left:     QuantityArray
    t_right:    QuantityArray
    fnu_net:  QuantityArray

    # @classmethod
    # def from_csv(
    #     cls,
    #     path:Path
    # )->Self:
    #     df = FileReader.table_from_csv(
    #         path=path,
    #     )
    #     metadata = ensure_string_keys(df.attrs)
    #     units = DATACLASS_CONVERTER.structure(
    #         metadata,
    #         UnitMetadata
    #     )
    #     t_center = QuantityArray(
    #             values = dataframe_processors.convert_ndarray(
    #                         df,Columns.T_CENTER
    #                     ),
    #             unit = units.t_unit
    #     )
    #     t_right = QuantityArray(
    #             values = dataframe_processors.convert_ndarray(
    #                         df,Columns.T_RIGHT
    #                     ),
    #             unit = units.t_unit
    #     )
    #     t_left = QuantityArray(
    #             values = dataframe_processors.convert_ndarray(
    #                         df,Columns.T_LEFT
    #                     ),
    #             unit = units.t_unit
    #     )
    #     fnu_transient = QuantityArray(
    #             values = dataframe_processors.convert_ndarray(
    #                         df,Columns.FNU_TRANSIENT
    #                     ),
    #             unit = units.fnu_unit
    #     )
    #     fnu_persistent = QuantityArray(
    #             values = dataframe_processors.convert_ndarray(
    #                         df,Columns.FNU_PERSISTENT
    #                     ),
    #             unit = units.fnu_unit
    #     )
    #     fnu_persistent_err = QuantityArray(
    #             values = dataframe_processors.convert_ndarray(
    #                         df,Columns.FNU_PERSISTENT_ERR
    #                     ),
    #             unit = units.fnu_unit
    #     )
    #     return cls(
    #         t_center=t_center,
    #         t_left = t_left,
    #         t_right = t_right,
    #         fnu_transient = fnu_transient,
    #         fnu_persistent = fnu_persistent,
    #         fnu_persistent_err = fnu_persistent_err
    #     )

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

    def fnu_total(
        self,
        fnu_persistent: QuantityArray,
        fnu_persistent_err: QuantityArray
    ):
        fnu_unit = self.fnu_net.unit
        fnu_trans = self.fnu_net.to_value_in(fnu_unit)
        fnu_per = fnu_persistent.to_value_in(fnu_unit)
        return QuantityArray(
            values = fnu_trans + fnu_per,
            unit = fnu_unit
        )

    def to_df(self,t_unit:str,fnu_unit:str):
        df = pd.DataFrame({
            Columns.T_LEFT: self.t_left.to_value_in(t_unit),
            Columns.T_CENTER: self.t_center.to_value_in(t_unit),
            Columns.T_RIGHT: self.t_right.to_value_in(t_unit),
            Columns.FNU_TRANSIENT: self.fnu_net.to_value_in(fnu_unit),
        })
        df.attrs[Columns.T_UNIT] = t_unit
        df.attrs[Columns.FNU_UNIT] = fnu_unit

        return df

    def plot(
        self,
        ax: Axes,
        style: LineStyle,
        *,
        fnu_persistent:QuantityArray,
        fnu_persistent_err:QuantityArray,
        t_unit: str,
        fnu_unit: str
    )->StepPatch:
        fnu_total = self.fnu_total(fnu_persistent=fnu_persistent,fnu_persistent_err=fnu_persistent_err).to_value_in(fnu_unit)
        bin_edges = self.bin_edges.to_value_in(t_unit)

        return ax.stairs(
            fnu_total,
            bin_edges,
            baseline=fnu_persistent.to_value_in(fnu_unit)[0],
            **style.to_kwargs(),
        )

@dataclass(frozen=True,slots=True)
class BinnedCalculationLightcurve:
    estimated_params: EstimatedParameters
    units: UnitMetadata
    t_center: FloatArray
    t_left: FloatArray
    t_right: FloatArray
    fnu: QuantityArray

    @property
    def bin_edges(self):
        return np.concatenate(
            (
                self.t_left,
                np.array([self.t_right[-1]]),
            )
        )

    @property
    def bin_edges_quantity(self):
        return QuantityArray(
            values=self.bin_edges,
            unit=self.units.t_unit
        )

    @property
    def t_left_quantity(self)->QuantityArray:
        return QuantityArray(
            values = self.t_left,
            unit = self.units.t_unit
        )

    @property
    def t_right_quantity(self)->QuantityArray:
        return QuantityArray(
            values = self.t_right,
            unit = self.units.t_unit
        )

    @property
    def t_center_quantity(self)->QuantityArray:
        return QuantityArray(
            values = self.t_center,
            unit = self.units.t_unit
        )

    def fnu_total(
        self,
        persistent: QuantityArray,
        persistent_err: QuantityArray
    )->QuantityArray:
        fnu_unit = self.units.fnu_unit
        fnu_net = self.fnu.to_value_in(fnu_unit)
        fnu_per = persistent.to_value_in(fnu_unit)
        fnu_per_err = persistent_err.to_value_in(fnu_unit)

        if not (
            self.t_center.size
            == fnu_per.size
            == fnu_per_err.size
        ):
            raise ValueError(
                "計算光度曲線と定常放射の配列の長さが一致しません."
            )

        return QuantityArray(
            values = fnu_net + fnu_per,
            unit = fnu_unit
        )

    def extract(
        self,
        idx:int
    )->BinnedBestfitLightcurve:
        params_best = self.estimated_params.extract_bestfit(idx)
        fnu_best = QuantityArray(
            values = self.fnu.values[idx],
            unit = self.fnu.unit
        )
        return BinnedBestfitLightcurve(
            params = params_best,
            fnu_net=fnu_best,
            t_center=self.t_center_quantity,
            t_right=self.t_right_quantity,
            t_left=self.t_left_quantity
        )

@dataclass(frozen=True, slots=True)
class CalculationLightcurve:
    params: EstimatedParameters
    t: QuantityArray
    fnu: QuantityArray

    def __post_init__(self)->None:
        t_arr = self.t.values
        fnu_grid = self.fnu.values

        if t_arr.ndim != 1:
            raise ValueError(
                "tは1次元配列でなければなりません. "
                f"shape={t_arr.shape}"
            )

        if fnu_grid.ndim != 2:
            raise ValueError(
                "fnuは2次元配列でなければなりません. "
                f"shape={fnu_grid.shape}"
            )

        num_times = t_arr.size
        num_flux_times = fnu_grid.shape[1]

        if num_times != num_flux_times:
            raise ValueError(
                "tの要素数とfnuの列数が一致していません. "
                f"t={t_arr.shape}, "
                f"fnu={fnu_grid.shape}"
            )

        return

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
        t_model: QuantityArray,
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
    def with_zero_boundary(
        self
    )->Self:
        """
            配列の左端に時刻ゼロでの境界条件を挿入する
        """
        t_values = self.t.values
        fnu_values = self.fnu.values

        if t_values[0] == 0.0:
            return self

        if t_values[0] < 0.0:
            raise ValueError(
                "計算光度曲線に負の時刻が含まれています."
                f"t_min={t_values[0]}"
            )


        return type(self)(
            params=self.params,
            t=QuantityArray(
                values=np.concatenate((
                    np.array([0.0], dtype=np.float64),
                    t_values,
                )),
                unit=self.t.unit,
            ),
            fnu=QuantityArray(
                values=np.concatenate((
                    np.zeros(
                        (fnu_values.shape[0], 1),
                        dtype=np.float64,
                    ),
                    fnu_values,
                ), axis=1),
                unit=self.fnu.unit,
            ),
        )

    def bin_average(
        self,
        bin_config: observation.Binning,
        drop_incomplete_bin: bool = True
    )->BinnedCalculationLightcurve:

        ## 計算光度曲線が観測の時間範囲を覆っているかcheck
        t = self.t
        fnu = self.fnu
        self.validate_time_coverage(t,bin_config)

        t_unit = bin_config.t_unit
        fnu_unit = fnu.unit

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
        ) -> FloatGrid:
            """
            全モデルについて、指定時刻で線形補間する

            Returns
            -------
            shape: (M, len(t_evaluate))
            """
            right_index = np.searchsorted(
                t_model,
                t_evaluate,
                side="left",
            )

            # t_modelの範囲内であることは上で検証済み
            right_index = np.clip(
                right_index,
                1,
                t_model.size - 1,
            )
            left_index = right_index - 1

            t0 = t_model[left_index]
            t1 = t_model[right_index]

            fraction = (
                (t_evaluate - t0)
                / (t1 - t0)
            )

            fnu0 = fnu_model[:, left_index]
            fnu1 = fnu_model[:, right_index]

            return (
                fnu0
                + (fnu1 - fnu0)
                * fraction[np.newaxis, :]
            )

        # shapeはいずれも(M, B)
        fnu_left = interpolate_flux(t_left)
        fnu_right = interpolate_flux(t_right)

        fnu_average_by_bin: list[FloatArray] = []

        for i, (left, right) in enumerate(
            zip(t_left, t_right, strict=True)
        ):
            inside = (
                (t_model > left)
                & (t_model < right)
            )

            # shape: (K_i,)
            t_bin = np.concatenate((
                np.array([left], dtype=np.float64),
                t_model[inside],
                np.array([right], dtype=np.float64),
            ))

            # shape: (M, K_i)
            fnu_bin = np.concatenate(
                (
                    fnu_left[:, i:i + 1],
                    fnu_model[:, inside],
                    fnu_right[:, i:i + 1],
                ),
                axis=1,
            )

            # 時間軸axis=1について積分
            # shape: (M,)
            flux_integral = np.trapezoid(
                fnu_bin,
                x=t_bin,
                axis=1,
            )

            fnu_average_by_bin.append(
                flux_integral / (right - left)
            )

        # (B個の(M,)配列) -> (M, B)
        fnu_average = QuantityArray(
            values = np.stack(
                fnu_average_by_bin,
                axis=1,
            ),
            unit = fnu_unit
        )

        return BinnedCalculationLightcurve(
            estimated_params=self.params,
            units=UnitMetadata(
                t_unit=t_unit,
                fnu_unit=str(fnu_unit),
            ),
            t_center=t_center,
            t_left=t_left,
            t_right=t_right,
            fnu=fnu_average,
        )

        # # interpolate value of fnu at bin-boundary
        # fnu_left: FloatArray = np.interp(
        #     bin_config.t_left,
        #     t_model,
        #     fnu_model,
        # )
        # fnu_right: FloatArray = np.interp(
        #     bin_config.t_right,
        #     t_model,
        #     fnu_model,
        # )
        #
        # fnu_average_list: list[float] = []
        #
        # for i, (left, right) in enumerate(
        #     zip(
        #         bin_config.t_left,
        #         bin_config.t_right,
        #         strict=True,
        #     )
        # ):
        #     inside = (
        #         (t_model > left)
        #         & (t_model < right)
        #     )
        #
        #     # create array for each bin
        #     t_bin: FloatArray = np.concatenate(
        #         (
        #             np.array(
        #                 [left],
        #                 dtype=np.float64
        #             ),
        #             t_model[inside],
        #             np.array(
        #                 [right],
        #                 dtype=np.float64
        #             ),
        #         )
        #     )
        #     fnu_bin: FloatArray = np.concatenate(
        #         (
        #             np.array(
        #                 [fnu_left[i]],
        #                 dtype=np.float64
        #             ),
        #             fnu_model[inside],
        #             np.array(
        #                 [fnu_right[i]],
        #                 dtype=np.float64
        #             ),
        #         )
        #     )
        #
        #     # integral within bin[i]
        #     flux_integral = Integrator.trapezoid(
        #         t_bin,
        #         fnu_bin
        #     )
        #
        #     fnu_average_list.append(
        #         flux_integral/ (right - left)
        #     )
        #
        # fnu_average: FloatArray = np.asarray(
        #     fnu_average_list,
        #     dtype=np.float64,
        # )
        #
        # return BinnedCalculationLightcurve(
        #     estimated_params=self.params,
        #     units = UnitMetadata(
        #         t_unit = t_unit,
        #         fnu_unit = str(fnu_unit),
        #     ),
        #     t_center= bin_config.t_center,
        #     t_left = bin_config.t_left,
        #     t_right = bin_config.t_right,
        #     fnu_averaged = fnu_average
        # )

def compute(
    config: inputs_as_dataclass.ModelConfigure,
    params: EstimatedParameters,
    integral_table : models.ThermalSynchrotronTable
)->CalculationLightcurve:
    t = config.time.quantity_array
    nu = QuantityData.from_Quantity(config.nu.quantity)
    phi = models.SynchrotronPhase.from_time_given_nu(
        t = t,
        nu = nu
    )
    xm = models.DimensionlessFrequency.from_phi(
        phi = phi,
        phi_theta = models.SynchrotronPhase(
            values = params.phi_theta.values,
            unit = params.phi_theta.unit
        )
    )
    lambda_theta = models.DimensionlessLuminosity.from_xm(
        xm = xm,
        tau_theta_array = params.tau_theta,
        integral_table = integral_table
    )
    lnu = lambda_theta.to_lnu(
        lnu_theta = params.lnu_theta
    )
    fnu_src = models.SourceframeSpecificFlux.from_lnu(
        lnu = lnu,
        distance = params.distance
    )
    fnu_obs = fnu_src.to_observerframe(
        doppler_factor = params.doppler_delta
    )
    lc = CalculationLightcurve(
        params = params,
        t = t,
        fnu = fnu_obs
    )
    return lc.with_zero_boundary()

def build_inputs(path:Path):
    df = FileReader.table_from_csv(path)
    lc_inputunits = InputUnits.from_keyvalue(path)
    return [
        Input(
            values=InputValues(**record),
            units=lc_inputunits
        )
        for record in df.to_dict("records")
    ]

@dataclass(frozen=True, slots=True)
class LightcurveStyleConfig:
    model_binned: plot_utils.LineStyle
    observation: plot_utils.ScatterStyle
    persistent: plot_utils.LineStyle

@dataclass(frozen=True, slots=True)
class PlotConfig:
    layout: plot_utils.PlotLayoutConfig
    styles: LightcurveStyleConfig

    @classmethod
    def from_yaml(
        cls,
        path: Path
    )->Self:
        dict_data = FileReader.yaml_safe(path)
        return DATACLASS_CONVERTER.structure(
            dict_data,
            cls
        )
