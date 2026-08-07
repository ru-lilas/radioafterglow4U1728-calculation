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

    def bin_average(
        self,
        binning: observation.Binning,
        drop_incomplete_bin: bool = True
    )->Self:
        if len(self.t_obs) != len(self.fnu_obs):
            raise ValueError(
                "t_obsとfnu_obsの配列長が一致しません."
            )
        if len(self.t_obs) < 2:
            raise ValueError(
                "t_obsの配列長が2より小さいためbin平均をとれません."
            )

        fnu_unit = cast(u.UnitBase,self.fnu_obs.unit)

        width = binning.bin_width
        t_arr: FloatArray = QuantityConverter.to_FloatArray(
            self.t_obs,
            binning.t_unit
        )
        fnu_value: FloatArray = QuantityConverter.to_FloatArray(self.fnu_obs,fnu_unit)

        duration = t_arr[-1] - t_arr[0]

        if width > duration:
            raise ValueError(
                "bin_widthは光度曲線の全時間範囲以下にしてください."
            )

        if np.any(np.diff(t_arr) <= 0.0):
            raise ValueError(
                "t_obsは単調増加でなければなりません."
            )

        t_edge = self._make_bin_edges(
            t_arr,
            width,
            drop_incomplete_bin
        )

        fnu_edge: FloatArray = np.interp(
            t_edge,
            t_arr,
            fnu_value,
        )

        flux_integral_list: list[float] = []

        for i, (left, right) in enumerate(
            zip(t_edge[:-1], t_edge[1:], strict=True)
        ):
            inside = (
                (t_arr > left)
                & (t_arr < right)
            )

            t_bin: FloatArray = np.concatenate(
                (
                    np.array([left], dtype=np.float64),
                    t_arr[inside],
                    np.array([right], dtype=np.float64),
                )
            )

            fnu_bin: FloatArray = np.concatenate(
                (
                    np.array([fnu_edge[i]], dtype=np.float64),
                    fnu_value[inside],
                    np.array([fnu_edge[i + 1]], dtype=np.float64),
                )
            )

            flux_integral_list.append(
                Integrator.trapezoid(t_bin, fnu_bin)
            )

        flux_integral: FloatArray = np.asarray(
            flux_integral_list,
            dtype=np.float64,
        )

        bin_widths: FloatArray = np.diff(t_edge)

        fnu_average: FloatArray = flux_integral / bin_widths

        t_bin_center: FloatArray = (
            t_edge[:-1] + t_edge[1:]
        ) / 2.0

        return type(self)(
            t_obs=t_bin_center * time_unit,
            fnu_obs=fnu_average * fnu_unit,
        )

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
    df = fr.read_csv(path)
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
