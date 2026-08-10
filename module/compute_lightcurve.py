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
from module.utils import Integrator
from module import observation
from module.parameter_table import Configure

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
        dict_data = fr.read_keyvalue(path)
        return from_dict(
            data_class = cls,
            data = dict_data
        )

@dataclass(frozen=True,slots=True)
class Input:
    values: InputValues
    units: InputUnits

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

class QuantityConverter:
    @staticmethod
    def to_scalar(quantity:u.Quantity,unit:str|u.UnitBase):
        quantity_arr:FloatArray = np.asarray(
            quantity.to_value(unit),
            dtype=np.float64
        )

        if quantity_arr.ndim != 0:
            raise ValueError("quantity はスカラーを指定してください.")

        return quantity_arr.item()

    @staticmethod
    def to_FloatArray(quantity:u.Quantity,unit:str|u.UnitBase):
        return np.asarray(
            quantity.to_value(quantity.unit),
            dtype=np.float64
        )

@dataclass(frozen=True, slots=True)
class CalculationLightcurve:
    t_observer_frame: u.Quantity
    fnu_observer_frame: u.Quantity

    def to_df(self,t_unit:str,fnu_unit:str):
        t_obs_value:FloatArray = np.asarray(
            self.t_observer_frame.to_value(t_unit),
            dtype=np.float64
        )
        fnu_obs_value:FloatArray = np.asarray(
            self.fnu_observer_frame.to_value(fnu_unit),
            dtype=np.float64
        )
        return pd.DataFrame({
            "t_obs": t_obs_value,
            "fnu_obs": fnu_obs_value
        })

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
        t_model: u.Quantity,
        binning: observation.Binning,
    ) -> None:
        t_model_value: FloatArray = np.asarray(
            t_model.to_value(binning.t_unit),
            dtype=np.float64,
        )

        if t_model_value.size < 2:
            raise ValueError(
                "理論光度曲線には2点以上の時刻が必要です."
            )

        if (
            binning.t_left[0] < t_model_value[0]
            and not np.isclose(
                binning.t_left[0],
                t_model[0],
            )
        ):
            raise ValueError(
                "理論光度曲線が最初の観測binを覆っていません."
            )

        if (
            binning.t_right[-1] > t_model_value[-1]
            and not np.isclose(
                binning.t_right[-1],
                t_model[-1],
            )
        ):
            raise ValueError(
                "理論光度曲線が最後の観測binを覆っていません."
            )

    def bin_average(
        self,
        binning: observation.Binning,
        drop_incomplete_bin: bool = True
    )->Self:
        self.validate_time_coverage(self.t_observer_frame,binning)

        t_unit = binning.t_unit
        fnu_unit = cast(u.UnitBase,self.fnu_observer_frame.unit)

        width = binning.bin_width
        t_model: FloatArray = QuantityConverter.to_FloatArray(
            self.t_observer_frame,
            binning.t_unit
        )
        fnu_model: FloatArray = QuantityConverter.to_FloatArray(
            self.fnu_observer_frame,
            fnu_unit
        )

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

        return type(self)(
            t_observer_frame=u.Quantity(
                binning.t_center,
                unit=t_unit,
            ),
            fnu_observer_frame=u.Quantity(
                fnu_average,
                unit=fnu_unit,
            ),
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
    time_unit = config.t_obs.unit
    fnu_unit = fnu_obs.unit

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
        t_observer_frame=u.Quantity(t_value,time_unit),
        fnu_observer_frame=u.Quantity(fnu_value,fnu_unit)
    )

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
