from collections.abc import Iterator,Mapping
from pathlib import Path
from dataclasses import dataclass
from typing import Self, cast, Any, Hashable
from dacite import from_dict
from numpy.typing import NDArray
from functools import cached_property
from module import plot_utils
from module.models import ThermalSynchrotron, ThermalSynchrotronTable, calculate_phi, calculate_xm
from module.mydataclasses import QuantityData,QuantityArray
from module.plot_utils import LineStyle
from module.types import FloatArray
from module.utilities import filereaders as fr
import astropy.units as u
import numpy as np
import pandas as pd
from module.utils import DataFrameUtils, Integrator,FileReader
from module import dataframe_processors, observation
from module.parameter_table import Configure
from enum import StrEnum, auto
from matplotlib.axes import Axes

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
        return from_dict(
            data_class = cls,
            data = dict_data
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
    t_unit: str|u.UnitBase
    fnu_unit: str|u.UnitBase

@dataclass(frozen=True,slots=True)
class BinnedCalculationLightcurve:
    units: UnitMetadata
    t_center: FloatArray
    t_left: FloatArray
    t_right: FloatArray
    fnu_averaged: FloatArray

    @classmethod
    def from_csv(
            cls,
            path:Path
    )->Self:
        dict_data = fr.read_keyvalue(path)
        df = FileReader.table_from_csv(
            path=path,
        )
        dict_data = df.attrs
        return cls(
            units = from_dict(data_class=UnitMetadata,data=dict_data),
            t_center =dataframe_processors.convert_ndarray(
                    df,Columns.T_CENTER
                ),
            t_right = dataframe_processors.convert_ndarray(
                df,Columns.T_RIGHT
            ),
            t_left = dataframe_processors.convert_ndarray(
                df,Columns.T_LEFT
            ),
            fnu_averaged = dataframe_processors.convert_ndarray(
                df,Columns.FNU_AVERAGED
            )
        )

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

    @property
    def fnu_averaged_quantity(self)->QuantityArray:
        return QuantityArray(
            values = self.fnu_averaged,
            unit = self.units.fnu_unit
        )

    def to_df(self,t_unit:str,fnu_unit:str):
        t_left = self.t_left_quantity
        t_center = self.t_center_quantity
        t_right = self.t_right_quantity
        fnu = self.fnu_averaged_quantity

        df = pd.DataFrame({
            Columns.T_LEFT: t_left.FloatArray_in(t_unit),
            Columns.T_CENTER: t_center.FloatArray_in(t_unit),
            Columns.T_RIGHT: t_right.FloatArray_in(t_unit),
            Columns.FNU_AVERAGED: fnu.FloatArray_in(fnu_unit)
        })
        df.attrs[Columns.T_UNIT] = t_unit
        df.attrs[Columns.FNU_UNIT] = fnu_unit

        return df

    def plot(
        self,
        ax: Axes,
        style: LineStyle,
        *,
        t_unit: str,
        fnu_unit: str
    ):
        t_left = self.t_left_quantity.FloatArray_in(t_unit)
        t_right = self.t_right_quantity.FloatArray_in(t_unit)
        fnu = self.fnu_averaged_quantity.FloatArray_in(fnu_unit)
        bin_edges = self.bin_edges_quantity.FloatArray_in(t_unit)

        return ax.stairs(
            fnu,
            bin_edges,
            **style.to_kwargs(),
        )


@dataclass(frozen=True, slots=True)
class CalculationLightcurve:
    t_observer_frame: QuantityArray
    fnu_observer_frame: QuantityArray

    @classmethod
    def from_csv(
            cls,
            path:Path
    )->Self:
        dict_data = fr.read_keyvalue(path)
        df = fr.read_csv(path)
        return cls(
            t_observer_frame = QuantityArray(
                values=dataframe_processors.convert_ndarray(
                    df,Columns.T_OBSERVER_FRAME
                ),
                unit=dict_data[Columns.T_UNIT]
            ),
            fnu_observer_frame = QuantityArray(
                values = dataframe_processors.convert_ndarray(
                    df,Columns.FNU_OBSERVER_FRAME
                ),
                unit=dict_data[Columns.FNU_UNIT]
            )
        )

    def to_df(self,t_unit:str,fnu_unit:str):
        df = pd.DataFrame({
            Columns.T_OBSERVER_FRAME: self.t_observer_frame.FloatArray_in(t_unit),
            Columns.FNU_OBSERVER_FRAME: self.fnu_observer_frame.FloatArray_in(fnu_unit)
        })
        df.attrs[Columns.T_UNIT] = t_unit
        df.attrs[Columns.FNU_UNIT] = fnu_unit
        return df


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

    def bin_average(
        self,
        binning: observation.Binning,
        drop_incomplete_bin: bool = True
    )->BinnedCalculationLightcurve:
        self.validate_time_coverage(self.t_observer_frame,binning)

        t_unit = binning.t_unit
        fnu_unit = cast(u.UnitBase,self.fnu_observer_frame.unit)

        width = binning.bin_width
        t_model = self.t_observer_frame.FloatArray_in(binning.t_unit)
        fnu_model = self.fnu_observer_frame.FloatArray_in(fnu_unit)

        if np.any(np.diff(t_model) <= 0.0):
            raise ValueError(
                "t_obsは単調増加でなければなりません."
            )

        num_bin = binning.t_center.size
        if not (
            binning.t_left.size
            == binning.t_right.size
            == num_bin
        ):
            raise ValueError(
                "観測binの配列長が一致しません."
            )

        # interpolate value of fnu at bin-boundary
        fnu_left: FloatArray = np.interp(
            binning.t_left,
            t_model,
            fnu_model,
        )
        fnu_right: FloatArray = np.interp(
            binning.t_right,
            t_model,
            fnu_model,
        )

        fnu_average_list: list[float] = []

        for i, (left, right) in enumerate(
            zip(
                binning.t_left,
                binning.t_right,
                strict=True,
            )
        ):
            inside = (
                (t_model > left)
                & (t_model < right)
            )

            # create array for each bin
            t_bin: FloatArray = np.concatenate(
                (
                    np.array(
                        [left],
                        dtype=np.float64
                    ),
                    t_model[inside],
                    np.array(
                        [right],
                        dtype=np.float64
                    ),
                )
            )
            fnu_bin: FloatArray = np.concatenate(
                (
                    np.array(
                        [fnu_left[i]],
                        dtype=np.float64
                    ),
                    fnu_model[inside],
                    np.array(
                        [fnu_right[i]],
                        dtype=np.float64
                    ),
                )
            )

            # integral within bin[i]
            flux_integral = Integrator.trapezoid(
                t_bin,
                fnu_bin
            )

            fnu_average_list.append(
                flux_integral/ (right - left)
            )

        fnu_average: FloatArray = np.asarray(
            fnu_average_list,
            dtype=np.float64,
        )

        # return type(self)(
        #     t_observer_frame=QuantityArray(
        #         binning.t_center,
        #         unit=t_unit,
        #     ),
        #     fnu_observer_frame=QuantityArray(
        #         fnu_average,
        #         unit=fnu_unit,
        #     ),
        # )
        return BinnedCalculationLightcurve(
            units = UnitMetadata(
                t_unit = t_unit,
                fnu_unit = fnu_unit
            ),
            t_center= binning.t_center,
            t_left = binning.t_left,
            t_right = binning.t_right,
            fnu_averaged = fnu_average
        )

def compute(
    config: Configure,
    model: ThermalSynchrotron,
    input: Input,
)->CalculationLightcurve:
    xm = calculate_xm(
        t_obs=config.t_obs,
        nu_obs=config.nu_obs,
        phi_theta=input.phi_theta
    )
    fnu_obs = model.fnu_obs(
        xm=xm,
        tau_theta=input.tau_theta,
        lnu_theta=input.lnu_theta,
        distance=input.distance,
        doppler_delta=input.doppler_delta
    )
    time_unit = str(config.t_obs.unit)
    fnu_unit = str(fnu_obs.unit)

    t_value: FloatArray = np.asarray(
        config.t_obs.to_value(time_unit),
        dtype=np.float64,
    )
    fnu_value: FloatArray = np.asarray(
        fnu_obs.to_value(fnu_unit),
        dtype=np.float64,
    )
    if t_value[0] > 0.0:
        t_value = np.concatenate(
            (
                np.array([0.0], dtype=np.float64),
                t_value,
            )
        )
        fnu_value = np.concatenate(
            (
                np.array([0.0], dtype=np.float64),
                fnu_value,
            )
        )
    return CalculationLightcurve(
        t_observer_frame=QuantityArray(
            values=t_value,
            unit=time_unit
        ),
        fnu_observer_frame=QuantityArray(
            fnu_value,
            fnu_unit
        )
    )



# def build_inputs(
#     path: Path,
# ) -> Iterator[tuple[int, Input]]:
#     df = fr.read_csv(path)
#     input_units = InputUnits.from_keyvalue(path)
#
#     for record in df.to_dict("records"):
#         idx = int(record["idx"])
#
#         yield (
#             idx,
#             Input(
#                 values=InputValues(**record),
#                 units=input_units,
#             ),
#         )
#
def build_inputs(path:Path):
    df = fr.read_csv(path)
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
        return from_dict(
            data_class = cls,
            data = dict_data
        )
