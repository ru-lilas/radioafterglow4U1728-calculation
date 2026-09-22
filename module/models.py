# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false

from functools import cached_property
from pathlib import Path
from typing import Self,Hashable,Any
import astropy.units as u
from matplotlib.axes import Axes
from module import plot_utils, quantity_converter
import numpy as np
import pandas as pd
from enum import StrEnum, auto
from dataclasses import dataclass
from module.mydataclasses import QuantityArray, QuantityData, QuantityArray1D
from module.types import FloatArray, FloatArray1D, FloatArrayLike, FloatGrid
from scipy import special
from numpy.typing import NDArray
from module import synchrotron_scaling_values
from module import electron_temperature
from module.utilities import bisection, unit_aliases as unit
from module.utils import FileReader
from astropy.constants import e,c,m_e,m_p,hbar
from module import f_factor
import astropy.units as u
e = u.Quantity(e.esu)
c = u.Quantity(c)
m_e = u.Quantity(m_e)
m_p = u.Quantity(m_p)
mec2 = (m_e*c**2).to("erg")

A_CYC = (((4.0*np.pi*mec2)/(3.0*e*c))**2).to("g cm-1")
PHI = (1*u.Unit("GHz")*1*u.Unit("min")).decompose()

MASS_RATIO_ELE_PRO = (m_e/m_p).decompose()

log10_2 = np.log10(2.0)
log10_3 = np.log10(3.0)
TWO_TO_FIVE:float = 2**5
THREE_TO_SEVEN = 3**7

class MetadataNames(StrEnum):
    PHI_TH = auto()
    THETA0 = auto()
    EPS_B = auto()
    EPS_TH = auto()
    MU_E_MU = auto()
    DISTANCE_VALUE = auto()
    DISTANCE_UNIT = auto()
    A_WIND_UNIT = auto()
    PHI_UNIT = auto()
    LNU_UNIT = auto()
    FNU_UNIT = auto()
    NU_UNIT = auto()

class ColumnNames(StrEnum):
    MODEL_INDEX = auto()
    PHI_TH = auto()
    THETA0 = auto()
    PHI_TH_THIN = auto()
    PHI_PEAK = auto()
    FNU_PEAK = auto()
    THETA = auto()
    THETA_THIN = auto()
    IS_SSA_THICK = auto()
    A_SYN = auto()
    A_WIND = auto()
    BETA_SH = auto()
    PHI_A = auto()
    PHI_THETA = auto()
    TAU_THETA = auto()
    LNU_THETA = auto()
    DOPPLER_DELTA = auto()
    TAU_TH_THIN = auto()
    LNU_TH_THIN = auto()
    PHI_PEAK_THIN = auto()
    FNU_PEAK_THIN = auto()
    LNU_PEAK_THIN = auto()
    NU = auto()

@dataclass(frozen=True,slots=True)
class ElectronTemperature:
    values:FloatArray

class MaxwellJuttnerFactor:

    @staticmethod
    def exact(
        theta:ElectronTemperature
    )->FloatArray:
        theta_val = theta.values
        return (
            2.0*theta_val**2 / special.kv(2,1.0/theta_val)
        )

    @staticmethod
    def hot(
        theta:ElectronTemperature
    )->FloatArray:
        theta_val = theta.values
        return np.ones_like(theta_val)

    def plot(
        self,
        theta:ElectronTemperature,
        ax:Axes,
        style:plot_utils.LineStyle,
    ):
        f_exact = self.exact(theta)
        return ax.plot(
            theta.values,f_exact,
            **style.to_kwargs()
        )

@dataclass(frozen=True,slots=True)
class PhaseParameter(QuantityArray):

    @classmethod
    def from_time_given_nu(
        cls,
        t: QuantityArray,
        nu: QuantityData
    )->Self:
        phi = t.quantity*nu.quantity
        return cls.from_quantity(phi)

    def to_time(
        self,
        nu: QuantityData
    )->QuantityArray:
        return QuantityArray.from_quantity(
            nu*self.quantity
        )

    def decompose(
        self
    )->FloatArray:
        return np.asarray(
            self.to_value_in("")
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
class EffectiveElectronProtonMassRatio: 
    value:float

    @classmethod
    def from_muep(
        cls,
        muep:float
    )->Self:
        return cls(
            value=float((muep*MASS_RATIO_ELE_PRO).decompose())
        )


@dataclass(frozen=True,slots=True)
class ElectronTemperatureScaling:
    values:FloatArray

    @classmethod
    def from_params(
        cls,
        eps_th:float,
        mep:float,
        beta_sh:FloatArray
    )->Self:
        return cls(
            values = 9.0*eps_th*beta_sh**2/(32.0*mep)
        )

    def to_beta(
        self,
        eps_th:float,
        mep:float,
    )->FloatArray:
        beta2 = (32.0*mep*self.values/(9.0*eps_th))
        return np.sqrt(beta2)

    def to_theta(
        self
    )->ElectronTemperature:
        return ElectronTemperature(
                values = DimensionlessElectronTemperature.hot(self.values)
        )

#=== A_wind ===#
class WindParameterNormalization:
    @staticmethod
    def cyclotron()->QuantityData:
        return QuantityData.from_Quantity(A_CYC)

    @staticmethod
    def synchrotron(
        theta:FloatArray
    )->QuantityArray:
        return QuantityArray.from_quantity(
            A_CYC/theta**4
        )

@dataclass(frozen=True,slots=True)
class NormalizedWindParameter(PhaseParameter):

    @classmethod
    def from_a_wind(
        cls,
        a_wind:QuantityArray,
        a_norm:QuantityArray
    )->Self:
        a_wind_q = a_wind.quantity
        a_norm_q = a_norm.quantity
        phi_a = np.sqrt((a_wind_q/a_norm_q))
        return cls.from_quantity(phi_a)

    def to_a_wind(
        self,
    )->QuantityArray:
        return QuantityArray.from_quantity(
            self.phi_a.quantity*self.a_norm.quantity
        )


#=== phi_theta ===#
@dataclass(frozen=True,slots=True)
class ThermalSynchrotronPhase(PhaseParameter):

    @classmethod
    def from_phiA_syn(
        cls,
        eps_b:float,
        phi_a:NormalizedWindParameter,
    )->Self:
        return cls.from_quantity(
            1.5*phi_a.quantity*np.sqrt(eps_b)
        )

    @classmethod
    def optically_thin_limit(
        cls,
        eps_b:float,
        eps_th:float,
        mep:float,
        theta:ElectronTemperature,
        theta0:ElectronTemperatureScaling,
    )->Self:
        theta_val = theta.values
        theta0_val = theta0.values
        f_factor = MaxwellJuttnerFactor.exact(theta)
        phi_th_thin_inv = u.Quantity(
                (np.pi*f_factor)/(eps_b*theta_val**7)
            *np.sqrt(TWO_TO_FIVE*eps_th*mep/(THREE_TO_SEVEN*theta0_val)),
            u.dimensionless_unscaled
        )
        return cls.from_quantity(phi_th_thin_inv**(-1))

    def to_phiA(
        self,
        eps_b:float,
    )->NormalizedWindParameter:
        return NormalizedWindParameter.from_quantity(
            2.0*self.quantity/(3.0*np.sqrt(eps_b))
        )

@dataclass(frozen=True,slots=True)
class WindParameter(QuantityArray):
    @classmethod
    def from_phi_th(
        cls,
        eps_b:float,
        a_syn:QuantityArray,
        phi_th:ThermalSynchrotronPhase
    )->Self:
        phi_val = phi_th.decompose()
        return cls.from_quantity(
            4.0*a_syn.quantity*phi_val**2/(9.0*eps_b)
        )


@dataclass(frozen=True,slots=True)
class ElectronTemperatureOpticallyThinLimit(ElectronTemperature):
    @classmethod
    def from_params(
        cls,
        mep:float,
        eps_b:float,
        phi_th:ThermalSynchrotronPhase
    )->Self:
        phith_val = phi_th.decompose()
        log10_phith = np.log10(phith_val)
        log10_mep = np.log10(mep)
        log10_epsb = np.log10(eps_b)
        log10_theta = np.asarray((
            2.0*log10_phith \
            -8.0*log10_3 \
            +3.0*log10_2 \
            -2.0*log10_epsb \
            +log10_mep
        )/15.0)
        theta_lin15_2 = 2.0*phith_val*np.sqrt(2.0*mep) / (81.0*eps_b)
        theta_lin = theta_lin15_2**(2.0/15.0)
        print(theta_lin)
        print(10**log10_theta)
        return cls(10**log10_theta)

class SelfAbsorptionOpticallyThinLimit:
    @staticmethod
    def theta0(
        eps_b:float,
        eps_th:float,
        phith:ThermalSynchrotronPhase,
        mep:float
    )->ElectronTemperatureOpticallyThinLimit:
        phi = phith.decompose()
        return ElectronTemperatureOpticallyThinLimit(
            (
            (np.pi*phi/eps_b)**2
                *2.0**5
                *3.0**7
                *eps_th
                *mep
        )**(1/15.0)
    )

@dataclass(frozen=True,slots=True)
class OpticalDepthParameter:
    values:FloatArray
    @classmethod
    def from_params_exact(
        cls,
        mep: float,
        eps_b:float,
        eps_th:float,
        theta:ElectronTemperature,
        theta0:ElectronTemperatureScaling,
        phi_th:ThermalSynchrotronPhase,
        hot:bool=False
    )->Self:
        phi = phi_th.decompose()
        f_factor = MaxwellJuttnerFactor.exact(theta) if not hot else MaxwellJuttnerFactor.hot(theta)
        return cls(
            values = (
                (np.pi*phi/(eps_b*theta.values**7))
                *np.sqrt(
                    32.0*eps_th*mep
                    / (3.0**7*theta0.values)
                )*f_factor
            )
        )

@dataclass(frozen=True,slots=True)
class ThermalSynchrotronLuminosity(QuantityArray):
    @classmethod
    def from_params(
        cls,
        mep: float,
        eps_th:float,
        theta:ElectronTemperature,
        theta0:ElectronTemperatureScaling,
        phi_th:ThermalSynchrotronPhase
    )->Self:
        phith_val = phi_th.decompose()
        quantities = (
            2.0**7*np.pi**2 / (9.0*eps_th)
            * mep
            * phith_val**2
            * theta0.values
            * theta.values
            * mec2
        )
        return cls.from_quantity(quantities)

@dataclass(frozen=True,slots=True)
class MolecularWeightRatio:
    value: float

    @classmethod
    def from_params(
        cls,
        mu:float,
        mu_e:float
    )->Self:
        return cls(mu_e/mu)

    @classmethod
    def from_mep(
        cls,
        mep:float
    )->float:
        return float((mep/MASS_RATIO_ELE_PRO).decompose())

    def to_mep(
        self
    )->float:
        return float((self.value*MASS_RATIO_ELE_PRO).decompose())

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
)->QuantityArray:
    return QuantityArray.from_quantity(
        192*np.pi**3*eps_b*mep*a**2*mec2/eps_th
    )

@dataclass(frozen=True,slots=True)
class ShockRadius(QuantityArray):

    @classmethod
    def beta_sh(
        cls,
        beta_sh:FloatArray,
        t:QuantityArray,
    ):
        return


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
        return cls.from_quantity(fnu)

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

@dataclass(frozen=True,slots=True)
class ModelMetadata:
    eps_b: float
    eps_th: float
    mue_mu: float
    distance: QuantityData
    phi_unit: str
    lnu_unit: str
    fnu_unit: str
    a_wind_unit: str

    @classmethod
    def from_attrs(
        cls,
        attrs:dict[Hashable,Any]
    )->Self:
        return cls(
            eps_b = attrs[MetadataNames.EPS_B],
            eps_th = attrs[MetadataNames.EPS_TH],
            mue_mu = attrs[MetadataNames.MU_E_MU],
            distance = QuantityData(
                value=attrs[MetadataNames.DISTANCE_VALUE],
                unit =attrs[MetadataNames.DISTANCE_UNIT]
            ),
            a_wind_unit = attrs[MetadataNames.A_WIND_UNIT],
            phi_unit = attrs[MetadataNames.PHI_UNIT],
            lnu_unit = attrs[MetadataNames.LNU_UNIT],
            fnu_unit = attrs[MetadataNames.FNU_UNIT],
        )
