# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false

from functools import cached_property
from pathlib import Path
from typing import Self
import astropy.units as u
from module import quantity_converter
import numpy as np
import pandas as pd
from dataclasses import dataclass
from module.mydataclasses import QuantityArray, QuantityData
from module.types import FloatArray, FloatArrayLike

from numpy.typing import NDArray
from module import synchrotron_scaling_values
from module import electron_temperature
from module import tabular
from module.utilities import bisection, unit_aliases as unit
from module import calculate_lnu_th
from module.utils import FileReader

# @dataclass
# class ThermalSynchrotronScalingValues:
#     input:InputParameters
#     nu_value: NDArray[np.float64]
#     nu_unit: str
#     table: tabular.ThermalSynchrotronTable
#
#     @cached_property
#     def nu(self)->u.Quantity:
#         return u.Quantity(self.nu_value,u.Unit(self.nu_unit))
#
#     @cached_property
#     def t_theta(self):
#         return synchrotron_scaling_values.calculate_t_theta(
#             phi_theta=self.input.phi_theta,
#             nu=self.nu
#         )
#
#     @cached_property
#     def r_theta(self):
#         return synchrotron_scaling_values.calculate_r_theta(
#             beta_sh=self.input.beta_sh,
#             t_theta=self.t_theta
#         )
#
#     @cached_property
#     def n_e_theta(self):
#         return synchrotron_scaling_values.calculate_n_ele_theta(
#             beta_sh=self.input.beta_sh,
#             mu=self.input.mu,
#             mu_e=self.input.mu_e,
#             a_wind = self.input.a_wind_quantity,
#             t_theta=self.t_theta
#         )
#
#     @cached_property
#     def b_mag_theta(self):
#         return synchrotron_scaling_values.calculate_b_mag_theta(
#             eps_B=self.input.eps_b,
#             a_wind=self.input.a_wind_quantity,
#             t_theta=self.t_theta
#         )
#
#     @cached_property
#     def j_theta(self):
#         return synchrotron_scaling_values.calculate_j_theta(self.input.theta,self.n_e_theta,self.b_mag_theta)
#
#     @cached_property
#     def alpha_theta(self):
#         return synchrotron_scaling_values.calculate_alpha_theta(self.input.theta,self.n_e_theta,self.b_mag_theta)
#
#     @cached_property
#     def tau_theta(self):
#         return self.input.tau_theta
#
#     @cached_property
#     def xi_est(self):
#         tau_theta = self.input.tau_theta
#         def f(xi:NDArray[np.float64]):
#             return synchrotron_scaling_values.calculate_ln_tau(xi,tau_theta,self.table)
#         xi_peak = bisection.bisection(f,1.0e-01,1.0e+04)
#         return np.asarray(xi_peak,dtype=np.float64)
#
#     @cached_property
#     def lnu_est_dimless(self):
#         lnu_peak_dimless = synchrotron_scaling_values.calculate_lambda_using_table(
#             tau_theta=self.input.tau_theta,
#             xm=self.xi_est,
#             table=self.table
#         )
#         return lnu_peak_dimless
#
#     @cached_property
#     def phi_peak(self):
#         phi_peak = u.Quantity(self.xi_est * self.input.phi_theta)
#         return phi_peak
#
#     @cached_property
#     def t_peak(self)->u.Quantity:
#         return u.Quantity(self.phi_peak/self.nu).to(self.t_theta.unit)
#
#     @cached_property
#     def lnu_peak(self):
#         lnu_peak = u.Quantity(self.l_theta*self.lnu_est_dimless,self.l_theta.unit)
#         return lnu_peak
#
# @dataclass
# class Frequency:
#     value_array: np.ndarray
#     unit: str
#
# @dataclass
# class SynchrotronSpectrum:
#     inputs: InputParameters
#     t_value: float
#     t_unit: str
#     nu_value: NDArray[np.float64]
#     nu_unit: str
#     tabular: tabular.ThermalSynchrotronTable
#
#     @cached_property
#     def scalings(self):
#         return ThermalSynchrotronScalingValues(
#             input=self.inputs,
#             nu_value=self.nu_value,
#             nu_unit=self.nu_unit,
#             table=self.tabular
#         )
#
#     @property
#     def t_quantity(self):
#         return u.Quantity(self.t_value,u.Unit(self.t_unit))
#
#     @property
#     def phi(self)->u.Quantity:
#         nu = self.scalings.nu
#         return nu*self.t_quantity
#
#     @property
#     def xi(self):
#         xi = np.asarray((self.phi/self.inputs.phi_theta).to_value(unit.dimensionless),dtype=np.float64)
#         return xi
#
#     @property
#     def ln_tau(self):
#         return synchrotron_scaling_values.calculate_ln_tau(
#             xi=self.xi,
#             tau_theta=self.scalings.tau_theta,
#             table=self.tabular
#         )
#
#     @property
#     def lnu_th_dimless(self):
#         return calculate_lnu_th.dimless_tabular(
#             x=self.xi,
#             log_tau=self.ln_tau
#         )
#
#     @property
#     def lnu_th(self):
#         q = u.Quantity((self.inputs.l_theta * self.lnu_th_dimless).to(self.scalings.l_theta.unit))
#         return q
#

##=== 2026/07/28 追加 ===##

@dataclass(frozen=True,slots=True)
class ThermalSynchrotronTable:
    xm: FloatArray
    ln_ip: FloatArray

    @classmethod
    def from_csv(
            cls,
            path:Path
    )-> Self:
        df = FileReader.table_from_csv(path)
        return cls(
            xm = df["xm"].to_numpy(dtype=np.float64),
            ln_ip = df["ln_ip"].to_numpy(dtype=np.float64)
        )

    def interpolate_log_ip(self,x:FloatArray)->FloatArray:
        return np.interp(x, self.xm, self.ln_ip)

    def interpolate_ip(self,x:FloatArray)->FloatArray:
        return np.exp(self.interpolate_log_ip(x))

@dataclass(slots=True)
class ThermalSynchrotron:
    integral_table: ThermalSynchrotronTable
    xm: FloatArray
    tau_theta: FloatArrayLike

    @property
    def lambda_theta(self)->FloatArray:
        ip = self.integral_table.interpolate_ip(self.xm)
        optical_depth = ip*self.tau_theta/self.xm
        f_esc = -np.expm1(-optical_depth)
        xm2 = self.xm**2
        return xm2 * f_esc

    @property
    def lambda_theta_max(self)->float:
        return float(np.argmax(self.lambda_theta))

class ThermalSynchrotronUtils:
    def __init__(
        self,
        table: ThermalSynchrotronTable,
    ) -> None:
        self._table = table

    def lambda_theta(
        self,
        xm:FloatArray,
        tau_theta:float|FloatArray,
    )->FloatArray:
        ip = self._table.interpolate_ip(xm)
        optical_depth = ip*tau_theta/xm
        f_esc = -np.expm1(-optical_depth)
        xm2 = xm**2
        return xm2 * f_esc

    def lnu(
        self,
        xm:FloatArray,
        tau_theta:float|FloatArray,
        lnu_theta: u.Quantity,
    )->u.Quantity:
        lambda_syn = self.lambda_theta(xm,tau_theta)
        return lnu_theta*lambda_syn
    
    def fnu_src(
        self,
        xm: FloatArray,
        tau_theta: float|FloatArray,
        lnu_theta: u.Quantity,
        distance: u.Quantity
    )->u.Quantity:
        lnu = self.lnu(xm,tau_theta,lnu_theta)
        return quantity_converter.lnu_into_fnu(lnu,distance)

    def fnu_obs(
        self,
        xm: FloatArray,
        tau_theta: float|FloatArray,
        lnu_theta: u.Quantity,
        distance: u.Quantity,
        doppler_delta: float|FloatArray
    )->u.Quantity:
        fnu_obs = self.fnu_src(xm,tau_theta,lnu_theta,distance)
        return doppler_delta**3 * fnu_obs

def lnu_into_fnu(
    lnu: QuantityArray,
    distance: QuantityData
)->QuantityArray:
    values:FloatArray = lnu.values / (4.0*np.pi*distance.value**2)
    unit:str = u.Unit(lnu.unit)/u.Unit(distance.unit)**2
    return QuantityArray(
        values = values,
        unit = unit
    )

def fnu_sourceframe_into_fnu_observerframe(
    fnu_sourceframe: QuantityArray,
    doppler_delta: FloatArrayLike
)->QuantityArray:
    return QuantityArray(
        values = fnu_sourceframe.values*doppler_delta**3,
        unit = fnu_sourceframe.unit
    )

def calculate_phi(
    t_obs: u.Quantity,
    nu_obs: u.Quantity
)->u.Quantity:
    """
    Args:
        t_obs: 観測者系での時間
        nu_obs: 観測者系での周波数
    """
    return t_obs*nu_obs

def calculate_xm(
    t_obs: u.Quantity,
    nu_obs: u.Quantity,
    phi_theta:u.Quantity
)->FloatArray:
    """
    Args:
        t_obs: 観測者系での時間
        nu_obs: 観測者系での周波数
    """
    phi = calculate_phi(t_obs,nu_obs)
    return np.asarray(
        (phi/phi_theta).to(u.dimensionless_unscaled),
        dtype=np.float64
    )

