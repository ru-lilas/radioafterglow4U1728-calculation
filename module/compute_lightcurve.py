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

    @cached_property
    def a_wind(self)->u.Quantity:
        return u.Quantity(self.values.a_wind,self.units.a_wind)

    @cached_property
    def phi_theta(self)->u.Quantity:
        return u.Quantity(self.values.phi_theta,self.units.phi_theta)

    @cached_property
    def lnu_theta(self)->u.Quantity:
        return u.Quantity(self.values.lnu_theta,self.units.lnu_theta)

    @cached_property
    def distance(self)->u.Quantity:
        return u.Quantity(self.values.distance,self.units.distance)

    @cached_property
    def tau_theta(self)->float:
        return self.values.tau_theta

    @cached_property
    def doppler_delta(self)->float:
        return self.values.doppler_delta

@dataclass(frozen=True,slots=True)
class Configure:
    time: QuantityArray
    fnu_unit: str
    nu: QuantityData

    @classmethod
    def from_yaml(
            cls,
            path:Path
    )->Self:
        dict_data = fr.read_yaml_pyyaml(path)
        return from_dict(
            data_class = cls,
            data = dict_data
        )

    @cached_property
    def t_obs(self)->u.Quantity:
        return u.Quantity(self.time.values.arr,self.time.unit)
    
    @cached_property
    def nu_obs(self)->u.Quantity:
        return u.Quantity(self.nu.value,self.nu.unit)

class QuantityConverter:
    @staticmethod
    def to_scalar(quantity:u.Quantity):
        unit = quantity.unit
        quantity_arr:FloatArray = np.asarray(
            quantity.to_value(unit),
            dtype=np.float64
        )

        if quantity_arr.ndim != 0:
            raise ValueError("quantity はスカラーを指定してください.")

        return quantity_arr.item()

    @staticmethod
    def to_FloatArray(quantity:u.Quantity):
        return np.asarray(
            quantity.to_value(quantity.unit),
            dtype=np.float64
        )

@dataclass(frozen=True, slots=True)
class Lightcurve:
    t_obs: u.Quantity
    fnu_obs: u.Quantity

    def to_df(self,t_unit:str,fnu_unit:str):
        t_obs_value:FloatArray = np.asarray(
            self.t_obs.to_value(t_unit),
            dtype=np.float64
        )
        fnu_obs_value:FloatArray = np.asarray(
            self.fnu_obs.to_value(fnu_unit),
            dtype=np.float64
        )
        return pd.DataFrame({
            "t_obs": t_obs_value,
            "fnu_obs": fnu_obs_value
        })

    def bin_average(
        self,
        bin_width: u.Quantity,
        *,
        t_min: u.Quantity | None = None,
        t_max: u.Quantity | None = None
    )->Self:
        if len(self.t_obs) != len(self.fnu_obs):
            raise ValueError(
                "t_obsとfnu_obsの配列長が一致しません."
            )
        if len(self.t_obs) < 2:
            raise ValueError(
                "t_obsの配列長が2より小さいためbin平均をとれません."
            )

        time_unit = cast(u.UnitBase,self.t_obs.unit)
        fnu_unit = cast(u.UnitBase,self.fnu_obs.unit)

        width = QuantityConverter.to_scalar(bin_width)

        if width <= 0.0:
            raise ValueError("bin_width は正の値にしてください.")

        time: FloatArray = QuantityConverter.to_FloatArray(self.t_obs)
        fnu: FloatArray = QuantityConverter.to_FloatArray(self.fnu_obs)

        if np.any(np.diff(time) <= 0.0):
            raise ValueError(
                "t_obsは単調増加でなければなりません."
            )

        start = (
            time[0]
            if t_min is None
            else QuantityConverter.to_scalar(t_min)
        )
        stop = (
            time[-1]
            if t_max is None
            else QuantityConverter.to_scalar(t_max)
        )

        if start < time[0] or stop > time[-1]:
            raise ValueError(
                "The requested bin range must lie within t_obs."
            )

        if stop <= start:
            raise ValueError("t_maxはt_minよりも大きい値にしてください.")

        bin_edges = np.arange(
            start,
            stop + width,
            width,
            dtype=np.float64,
        )

        # 最後のbinが計算範囲を超える場合は除外する
        bin_edges = bin_edges[bin_edges <= stop]

        if bin_edges.size == 0 or bin_edges[-1] < stop:
            bin_edges = np.append(bin_edges, stop)

        binned_time: list[float] = []
        binned_flux: list[float] = []

        for left, right in zip(
            bin_edges[:-1],
            bin_edges[1:],
            strict=True,
        ):
            inside = (time > left) & (time < right)

            time_bin = np.concatenate(
                (
                    np.array([left]),
                    time[inside],
                    np.array([right]),
                )
            )

            flux_bin = np.interp(
                time_bin,
                time,
                fnu,
            )

            mean_flux = np.trapezoid(
                flux_bin,
                time_bin,
            ) / (right - left)

            binned_time.append(0.5 * (left + right))
            binned_flux.append(float(mean_flux))

        return type(self)(
            t_obs=np.asarray(
                binned_time,
                dtype=np.float64,
            ) * time_unit,
            fnu_obs=np.asarray(
                binned_flux,
                dtype=np.float64,
            ) * fnu_unit,
        )

        return

def compute(
    config: Configure,
    model: ThermalSynchrotron,
    input: Input,
)->Lightcurve:
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
    return Lightcurve(
        t_obs=config.t_obs,
        fnu_obs=fnu_obs
    )

def build_inputs(path:Path):
    df = fr.read_csv_within_idx(path)
    lc_inputunits = InputUnits.from_keyvalue(path)
    return [
        Input(
            values=InputValues(**record),
            units=lc_inputunits
        )
        for record in df.to_dict("records")
    ]

# @dataclass
# class LightcurveCalculation:
#     t: QuantityData
#     nu: QuantityData
#     table_integral:ThermalSynchrotronTable
#
#     @cached_property
#     def t_quantity(self):
#         return self.t.quantity
#
#     @cached_property
#     def nu_quantity(self):
#         return self.nu.quantity
#
#     def phi_arr(self,phi_theta:QuantityData):
#         phi = u.Quantity(self.t_quantity*self.nu_quantity)
#         return u.Quantity(phi.to(phi_theta.unit))
#
#     def xm_arr(
#         self,
#         phi_theta:QuantityData,
#     ):
#         xm = (self.phi_arr(phi_theta)/phi_theta.quantity).to_value(u.dimensionless_unscaled)
#         return np.asarray(xm,dtype=np.float64)
#
#     def lambda_arr(
#         self,
#         phi_theta:QuantityData,
#         tau_theta: NDArray[np.float64]
#     ):
#         return calculate_lambda_using_table(
#             xm=self.xm_arr(phi_theta),
#             tau_theta=tau_theta,
#             table=self.table_integral
#         )
#
#     def lnu_arr(
#         self,
#         phi_theta:QuantityData,
#         lnu_theta: QuantityData,
#         tau_theta: NDArray[np.float64]
#     ):
#         lnu =  u.Quantity(
#             self.lambda_arr(phi_theta,tau_theta)
#                 *lnu_theta.quantity
#         )
#         lnu_value = np.asarray(
#             lnu.to_value(lnu_theta.unit),dtype=np.float64
#         )
#         return QuantityData(lnu_value,lnu_theta.unit)
#
#     def fnu(
#         self,
#         phi_theta:QuantityData,
#         lnu_theta: QuantityData,
#         tau_theta: float,
#         d_src: QuantityData
#     ):
#         tau_theta_arr = np.asarray(tau_theta,dtype=np.float64)
#         lnu = self.lnu_arr(phi_theta,lnu_theta,tau_theta_arr)
#         fnu = quantity_converter.lnu_into_fnu(
#             lnu=lnu.quantity,
#             distance=d_src.quantity
#         )
#         fnu_value = np.asarray(
#             fnu.value,dtype=np.float64
#         )
#         fnu_unit = cast(u.Unit,fnu.unit).to_string()
#         return QuantityData(fnu_value,fnu_unit)
#
#     def fnu_with_doppler(
#         self,
#         phi_theta:QuantityData,
#         lnu_theta: QuantityData,
#         tau_theta: float,
#         d_src: QuantityData,
#         doppler_delta: float
#     ):
#         fnu_no_doppler = self.fnu(
#             phi_theta,lnu_theta,tau_theta,d_src
#         )
#         fnu_with_doppler = doppler_delta**3 *fnu_no_doppler.value
#         return QuantityData(
#             value = fnu_with_doppler,
#             unit = fnu_no_doppler.unit
#         )
