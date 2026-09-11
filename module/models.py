# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false

from functools import cached_property
from pathlib import Path
from typing import Self
import astropy.units as u
from matplotlib.axes import Axes
from module import plot_utils, quantity_converter
import numpy as np
import pandas as pd
from dataclasses import dataclass
from module.mydataclasses import QuantityArray, QuantityData, QuantityArray1D
from module.types import FloatArray, FloatArray1D, FloatArrayLike, FloatGrid
from scipy import special
from numpy.typing import NDArray
from module import synchrotron_scaling_values
from module import electron_temperature
from module import tabular
from module.utilities import bisection, unit_aliases as unit
from module import calculate_lnu_th
from module.utils import FileReader
from astropy.constants import e,c,m_e,m_p,hbar
from module import f_factor
import astropy.units as u
e = u.Quantity(e.esu)
c = u.Quantity(c)
m_e = u.Quantity(m_e)
m_p = u.Quantity(m_p)
hbar = u.Quantity(hbar)
mec2 = (m_e*c**2).to("erg")

A_CYC = (((4.0*np.pi*mec2)/(3.0*e*c))**2).to("g cm-1")
T_ELE = (hbar/mec2).to("s")
PHI = (1*u.Unit("GHz")*1*u.Unit("min")).decompose()

MASS_RATIO_ELE_PRO = (m_e/m_p).decompose()

class MaxwellJuttnerFactor:

    @staticmethod
    def exact(
        theta:FloatArray
    )->FloatArray:
        return (
            2.0*theta**2 / special.kv(2,1.0/theta)
        )

    @staticmethod
    def hot(
        theta:FloatArray
    )->FloatArray:
        return np.ones_like(theta)

    def plot(
        self,
        theta:FloatArray,
        ax:Axes,
        style:plot_utils.LineStyle,
    ):
        f_exact = self.exact(theta)
        ax.plot(
            theta,f_exact,
            **style.to_kwargs()
        )

@dataclass(frozen=True,slots=True)
class PhaseParameter(QuantityArray1D):

    @classmethod
    def from_time_given_nu(
        cls,
        t: QuantityArray,
        nu: QuantityData
    )->Self:
        phi = t.quantity*nu.quantity
        return cls.from_Quantity(phi)

    def to_time(
        self,
        nu: QuantityData
    )->QuantityArray:
        return QuantityArray.from_Quantity(
            nu*self.quantity
        )


class DimensionlessElectronTemperature:
    @staticmethod
    def gp98(
        theta0:FloatArray
    )->FloatArray:
        """
            Gammie&Popham(1998)
        """
        return np.asarray(
            (
                5.0 * theta0
                - 6.0
                + np.sqrt(
                    25.0 * theta0**2
                    + 180.0 * theta0
                    + 36.0
                )
            )
            / 30.0,
            dtype=np.float64,
        )
    @staticmethod
    def hot(
        theta0:FloatArray
    )->FloatArray:
        return theta0 / 3.0

@dataclass(frozen=True,slots=True)
class MolecularWeight:
    mu: float
    mu_e: float

    @property
    def effective_mass_ratio(self):
        return float((self.mu_e*MASS_RATIO_ELE_PRO/self.mu).decompose())

@dataclass(frozen=True,slots=True)
class ElectronTemperatureScaling:
    values:FloatArray

    @classmethod
    def from_params(
        cls,
        eps_th:float,
        m_ratio_ep:float,
        beta_sh:FloatArray
    )->Self:
        return cls(
            values = 9.0*eps_th*beta_sh**2/(32.0*m_ratio_ep)
        )

    def to_beta(
        self,
        eps_th:float,
        m_ratio_ep:float,
    )->FloatArray:
        beta2 = (32.0*m_ratio_ep*self.theta0/(9.0*eps_th))
        return np.sqrt(beta2)

    def to_theta(
        self
    )->FloatArray:
        return DimensionlessElectronTemperature.hot(self.theta0)

@dataclass(frozen=True,slots=True)
class NormalizedWindParameter(QuantityArray1D):

    @classmethod
    def from_a_wind(
        cls,
        a_wind:QuantityArray1D,
        a_norm:QuantityData
    )->Self:
        a_wind_q = a_wind.quantity
        a_norm_q = a_norm.quantity
        phi_a = np.sqrt((a_wind_q/a_norm_q)).to("GHz min")
        return cls.from_Quantity(phi_a)

    def to_a_wind(
        self,
        a_norm:QuantityData
    )->QuantityArray1D:
        return QuantityArray1D.from_Quantity(
            self.quantity*a_norm.quantity
        )

    def decompose(
        self
    )->FloatArray:
        return np.asarray(
            self.quantity.to_value(u.dimensionless_unscaled)
        )

@dataclass(frozen=True,slots=True)
class ThemalSynchrotronPhase(PhaseParameter):

    @classmethod
    def from_params(
        cls,
        eps_b:float,
        theta:FloatArray,
        phi_a:NormalizedWindParameter,
    )->Self:
        return cls.from_Quantity(
            1.5*theta**2*phi_a.quantity*np.sqrt(eps_b)
        )

@dataclass(frozen=True,slots=True)
class OpticalDepthParameter:
    values:FloatArray
    @classmethod
    def from_params(
        cls,
        mep: float,
        eps_b:float,
        theta:FloatArray,
        phi_a:NormalizedWindParameter
    )->Self:
        phi_a_decomposed = phi_a.decompose()
        return cls(
            values = (phi_a_decomposed/27.0)*np.sqrt(2.0*mep/(eps_b*theta**11)*MaxwellJuttnerFactor.exact(theta))
        )

def calculate_effective_mass_ratio(
    mu:float,
    mu_e:float
)->float:
    return float((mu_e*MASS_RATIO_ELE_PRO/mu).decompose())

def calculate_lnu_theta(
    eps_th:float,
    eps_b:float,
    mep:float,
    a:FloatArray
)->QuantityArray1D:
    return QuantityArray1D.from_Quantity(
        192*np.pi**3*eps_b*mep*a**2*mec2/eps_th
    )

@dataclass(frozen=True,slots=True)
class TransRelativisticLimit:
    mep:float
    eps_th: float
    eps_b: float

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

@dataclass(frozen=True,slots=True)
class DimensionlessFrequency:
    values: FloatArray

    @classmethod
    def from_phi(
        cls,
        phi:PhaseParameter,
        phi_theta:PhaseParameter
    )->Self:
        if phi.values.ndim != 1:
            raise ValueError(
                "phiは1次元配列でなければなりません. "
                f"shape={phi.values.shape}"
            )

        if phi_theta.values.ndim != 1:
            raise ValueError(
                "phi_thetaは1次元配列でなければなりません. "
                f"shape={phi_theta.values.shape}"
            )
        phi_row = phi.quantity[np.newaxis, :]
        phi_theta_column = phi_theta.quantity[:, np.newaxis]
        xm = (phi_row/phi_theta_column).to(u.dimensionless_unscaled)
        return cls(
            values = np.asarray(
                xm,
                dtype=np.float64
        ))

@dataclass(frozen=True,slots=True)
class DimensionlessLuminosity:
    values:FloatGrid

    @classmethod
    def from_xm(
        cls,
        xm: DimensionlessFrequency,
        tau_theta_array: FloatArray,
        integral_table: ThermalSynchrotronTable
    )->Self:
        xm_grid = xm.values
        if xm_grid.ndim != 2:
            raise ValueError(
                "xmは2次元配列でなければなりません. "
                f"shape={xm.values.shape}"
            )
        if tau_theta_array.ndim != 1:
            raise ValueError(
                "tau_thetaは1次元配列でなければなりません. "
                f"shape={tau_theta_array.shape}"
            )

        num_models = xm_grid.shape[0]

        if tau_theta_array.size != num_models:
            raise ValueError(
                "xmのモデル数とtau_thetaの要素数が"
                "一致していません. "
                f"xm={xm_grid.shape}, "
                f"tau_theta={tau_theta_array.shape}"
            )

        tau_theta_column = tau_theta_array[:,np.newaxis]
        ip_grid = integral_table.interpolate_ip(xm_grid)
        optical_depth = (
                ip_grid
                *tau_theta_column
                /xm.values
        )
        escape_fraction = -np.expm1(-optical_depth)
        xm2 = (xm.values)**2
        return cls(
            values = xm2 * escape_fraction
        )

    def to_lnu(
        self,
        lnu_theta:QuantityArray
    )->QuantityArray:
        lnu_theta_values = lnu_theta.values

        if lnu_theta_values.ndim != 1:
            raise ValueError(
                "lnu_thetaは1次元配列でなければなりません. "
                f"shape={lnu_theta_values.shape}"
            )

        if lnu_theta_values.size != self.values.shape[0]:
            raise ValueError(
                "無次元光度のモデル数とlnu_thetaの要素数が"
                "一致していません. "
                f"luminosity={self.values.shape}, "
                f"lnu_theta={lnu_theta_values.shape}"
            )

        lnu_theta_column = lnu_theta_values[:, np.newaxis]

        return QuantityArray(
            values=self.values * lnu_theta_column,
            unit=lnu_theta.unit,
        )

@dataclass(frozen=True,slots=True)
class ObserverframeSpecificFlux(QuantityArray):
    pass

@dataclass(frozen=True,slots=True)
class SourceframeSpecificFlux(QuantityArray):
    @classmethod
    def from_lnu(
        cls,
        lnu: QuantityArray,
        distance: QuantityData
    )->Self:
        fnu = quantity_converter.lnu_into_fnu(
            lnu = lnu.quantity,
            distance = distance.quantity
        )
        return cls.from_Quantity(fnu)

    def to_observerframe(
        self,
        doppler_factor:FloatArray
    )->ObserverframeSpecificFlux:
        doppler_factor_column = doppler_factor[:,np.newaxis]
        return ObserverframeSpecificFlux(
            values = doppler_factor_column**3 * self.values,
            unit = self.unit
        )

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


def theta(
    eps_th:FloatArrayLike,
    mu: FloatArrayLike,
    mu_e: FloatArrayLike,
    beta_sh: FloatArrayLike,
)->FloatArray:
    return electron_temperature.calculate_theta_e(
        eps_th=eps_th,
        mu=mu,
        mu_e=mu_e,
        beta_sh=beta_sh
    )

def phi_theta(
    theta:FloatArrayLike,
    eps_b:FloatArrayLike,
    a_wind: QuantityArray
):
    return synchrotron_scaling_values.calculate_phi_theta(
        theta = theta,
        eps_b = eps_b,
        a_wind = a_wind.quantity
    )

def tau_theta(
    theta:FloatArrayLike,
    eps_b:FloatArrayLike,
    a_wind: QuantityArray,
    beta_sh:FloatArrayLike,
    mu:FloatArrayLike,
    mu_e:FloatArrayLike
):
    return synchrotron_scaling_values.calculate_tau_theta(
        theta = theta,
        eps_b = eps_b,
        a_wind = a_wind.quantity,
        beta_sh=beta_sh,
        mu=mu,
        mu_e=mu_e,
    )

def lnu_theta(
    theta:FloatArrayLike,
    eps_b:FloatArrayLike,
    a_wind: QuantityArray,
    beta_sh:FloatArrayLike,
):
    return synchrotron_scaling_values.calculate_l_theta(
        theta = theta,
        eps_b = eps_b,
        a_wind =a_wind.quantity,
        beta_sh=beta_sh,
    )

def doppler_delta(
    beta_sh:FloatArrayLike
):
    return quantity_converter.beta_into_doppler_delta(beta_sh)
