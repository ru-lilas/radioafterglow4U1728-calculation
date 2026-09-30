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
from module.mydataclasses import (
    FloatVector, FloatVectorSpace,
    NDArraySpace, QuantityArray,
    QuantityArrayBase, QuantityData,
    QuantityArray1D, QuantityGrid,
    ValueArray,IndexGrid
)
from module.synchrotron_numerical_table import ThermalSynchrotronIntegrals
from module.types import FloatArray, FloatArray1D, FloatArrayLike, FloatGrid
from scipy import special
from numpy.typing import NDArray
from module import electron_temperature
from module.utilities import bisection, unit_aliases as unit
from module.utils import FileReader
from astropy.constants import e,c,m_e,m_p
import astropy.units as u
e = u.Quantity(e.esu)
c = u.Quantity(c)
m_e = u.Quantity(m_e)
m_p = u.Quantity(m_p)
mec2 = (m_e*c**2).to("erg")

type NDArray64 = NDArray[np.float64]

A_CYC = (((4.0*np.pi*mec2)/(3.0*e*c))**2).to("g cm-1")
PHI = (1*u.Unit("GHz")*1*u.Unit("min")).decompose()

MASS_RATIO_ELE_PRO = (m_e/m_p).decompose()

log10_2 = np.log10(2.0)
log10_3 = np.log10(3.0)
TWO_TO_FIVE:float = 2**5
THREE_TO_SEVEN = 3**7

class DefaultUnits(StrEnum):
    fnu = "mJy"
    lnu = "erg s-1 Hz-1"
    t = "min"
    nu = "GHz"
    phi = "GHz min"

class MetadataNames(StrEnum):
    PHI_TH = auto()
    THETA0 = auto()
    GRID_NROWS = auto()
    GRID_NCOLS = auto()
    EPS_B = auto()
    EPS_TH = auto()
    MU_E = auto()
    MU = auto()
    P_INDEX = auto()
    EPS_E = auto()
    MU_E_MU = auto()
    DISTANCE_VALUE = auto()
    DISTANCE_UNIT = auto()
    A_WIND_UNIT = auto()
    PHI_UNIT = auto()
    LNU_UNIT = auto()
    FNU_UNIT = auto()
    NU_UNIT = auto()
    T_UNIT = auto()

class FluxName(StrEnum):
    FNU_THERMAL = auto()
    FNU_POWERLAW = auto()
    FNU_TOTAL = auto()

class PhiName(StrEnum):
    PHI = auto()

class ColumnNames(StrEnum):
    CANDIDATE_MODEL_INDEX = auto()
    MODEL_INDEX = auto()
    PEAK_COLUMN_INDEX = auto()
    XM = auto()
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
    T = auto()
    T_CENTER = auto()
    T_LEFT = auto()
    T_RIGHT = auto()
    FNU_OBS_FRAME = auto()
    LNU = auto()
    FNU_SRC_FRAME = auto()
    FNU_FLARE = auto()
    FNU_TOTAL = auto()
    FNU_BG = auto()

class ChiSquaredParameters(StrEnum):
    CHI_SQUARED = auto()
    N_DOF = auto()

@dataclass(frozen=True,slots=True)
class ScenarioParameter:
    eps_b: QuantityData
    eps_th: QuantityData
    mu_e_mu: QuantityData
    distance: QuantityData

@dataclass(frozen=True,slots=True)
class ElectronTemperature:
    values:FloatArray

class MaxwellJuttnerFactor:

    @staticmethod
    def exact(
        theta:NDArray[np.float64]
    )->NDArray[np.float64]:
        return (
            2.0*theta**2 / special.kv(2,1.0/theta)
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
        f_exact = self.exact(theta.values)
        return ax.plot(
            theta.values,f_exact,
            **style.to_kwargs()
        )

class PowerlawDistributionFactor:
    @staticmethod
    def exact(
        p: float,
        theta:NDArray[np.float64],
        gamma_min: NDArray[np.float64]
    )->NDArray[np.float64]:
        pm1 = p-1.0
        return (
            pm1*gamma_min/(pm1*gamma_min-pm1+1.0)
        )*(gamma_min/(3.0*theta))**pm1

    @staticmethod
    def hot(
        theta:NDArray[np.float64]
    )->FloatArray:
        return np.ones_like(theta)

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
        f_factor = MaxwellJuttnerFactor.exact(theta.values)
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
        f_factor = MaxwellJuttnerFactor.exact(theta.values) \
        if not hot else MaxwellJuttnerFactor.hot(theta)
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
class MolecularWeight:
    mu_e: float
    mu: float

    def to_ratio(
        self
    )->float:
        return float(
            (self.mu_e/self.mu).decompose()
        )

    def to_effective_mass_ratio(
        self
    )->float:
        return float(
            (self.mu_e*MASS_RATIO_ELE_PRO/self.mu).decompose()
        )

    @staticmethod
    def convert_mass_ratio_into_ratio(
        mass_ratio:float
    )->float:
        return float(mass_ratio/MASS_RATIO_ELE_PRO)

@dataclass(frozen=True,slots=True)
class PowerlawParameters:
    p_index:float
    eps_e: float

    @property
    def cj(
        self
    )->float:
        p=self.p_index
        return (
            3.0**(0.5*(2.0*p-1.0)) \
            *(p-2.0) \
            *special.gamma( 0.25*(p+5.0)) \
            *special.gamma( (3.0*p +  19.0)  / 12.0) \
            *special.gamma( (3.0*p -  1.0)   / 12.0) \
            / (
                2.0**(0.5*(7.0-p)) \
                *np.sqrt(np.pi) \
                *(p + 1.0) \
                *special.gamma(0.25*(p+7.0))
            )
        )

    @property
    def ca(
        self
    )->float:
        p=self.p_index
        return (
            2.0**(0.5*p) \
            *np.pi**1.5 \
            *(p-2) \
            *special.gamma( 0.25*(p+6.0)) \
            *special.gamma( (3.0*p +  2.0)  / 12.0) \
            *special.gamma( (3.0*p + 22.0)  / 12.0) \
            / (
                3.0**(0.5*(5.0-2.0*p)) \
                *special.gamma(0.25*(p+8.0))
            )
        )

@dataclass(frozen=True,slots=True)
class ParameterGrid:
    phi_th: NDArray64
    theta0: NDArray64

@dataclass(frozen=True,slots=True)
class InputParametersRaw:
    phi_th: QuantityArrayBase
    theta0: FloatVectorSpace
    eps_th: float
    eps_b: float
    molecular_weight: MolecularWeight
    distance: QuantityData
    powerlaw:PowerlawParameters

    @property
    def phi_arr(self)->FloatVector:
        return self.phi_th.to_floatvector()

    @property
    def theta0_arr(self)->FloatVector:
        return self.theta0.to_floatvector()

    def grid(self)->ParameterGrid:
        phi_axis = np.unique(self.phi_arr.values)
        theta0_axis = np.unique(self.theta0_arr.values)

        phi_grid,theta0_grid = np.meshgrid(
            phi_axis,
            theta0_axis,
            indexing="xy",
            sparse=False
        )

        return ParameterGrid(
            phi_th=phi_grid,
            theta0=theta0_grid
        )

@dataclass(frozen=True,slots=True)
class ModelParameters:
    phi_th: NDArray64
    theta0: NDArray64
    eps_th: float
    eps_b: float
    molecular_weight: MolecularWeight
    powerlaw:PowerlawParameters
    distance:QuantityData
    model_index: NDArray[np.int64] | None = None

    def __post_init__(self) -> None:
        if self.phi_th.ndim != 2 or self.phi_th.shape != self.theta0.shape:
            raise ValueError("phi_thとtheta0は同じ形状の2次元配列が必要です.")
        model_index = (
            np.arange(self.phi_th.size, dtype=np.int64).reshape(self.phi_th.shape)
            if self.model_index is None else np.asarray(self.model_index)
        )
        if model_index.shape != self.phi_th.shape or model_index.dtype.kind not in "iu":
            raise ValueError("model_indexはモデル格子と同じ形状の整数配列が必要です.")
        if not pd.Index(model_index.ravel()).is_unique:
            raise ValueError("model_indexに重複があってはなりません.")
        object.__setattr__(self, "model_index", model_index.astype(np.int64))

    @property
    def model_indices(self) -> NDArray[np.int64]:
        if self.model_index is None:
            raise RuntimeError("model_indexが初期化されていません.")
        return self.model_index

    @classmethod
    def build_from_input(
        cls,
        input:InputParametersRaw
    )->Self:
        grid=input.grid()
        return cls(
            phi_th=grid.phi_th,
            theta0=grid.theta0,
            eps_th=input.eps_th,
            eps_b =input.eps_b,
            molecular_weight=input.molecular_weight,
            distance=input.distance,
            powerlaw=input.powerlaw,

        )

    def to_df(self) -> pd.DataFrame:
        if self.phi_th.shape != self.theta0.shape or self.phi_th.ndim != 2:
            raise ValueError("phi_thとtheta0は同じ形状の2次元配列が必要です.")

        df = pd.DataFrame({
            ColumnNames.MODEL_INDEX: self.model_indices.ravel(),
            ColumnNames.PHI_TH: self.phi_th.ravel(),
            ColumnNames.THETA0: self.theta0.ravel(),
        })
        df.attrs = {
            MetadataNames.GRID_NROWS: self.phi_th.shape[0],
            MetadataNames.GRID_NCOLS: self.phi_th.shape[1],
            MetadataNames.EPS_TH: float(self.eps_th),
            MetadataNames.EPS_B: float(self.eps_b),
            MetadataNames.MU_E: float(self.molecular_weight.mu_e),
            MetadataNames.MU: float(self.molecular_weight.mu),
            MetadataNames.P_INDEX: float(self.powerlaw.p_index),
            MetadataNames.EPS_E: float(self.powerlaw.eps_e),
            MetadataNames.DISTANCE_VALUE: float(self.distance.value),
            MetadataNames.DISTANCE_UNIT: self.distance.unit,
        }
        return df

    def select_models(self, model_indices: list[int]) -> Self:
        """to_df()のmodel_indexで指定したモデルを入力順に取り出す."""
        indices = np.asarray(model_indices)
        if indices.ndim != 1 or indices.dtype.kind not in "iu" or indices.size == 0:
            raise ValueError("model_indicesには空でない整数のlistが必要です.")
        if not pd.Index(indices).is_unique:
            raise ValueError("model_indicesに重複があってはなりません.")
        positions = pd.Index(self.model_indices.ravel()).get_indexer(indices)
        if np.any(positions < 0):
            raise KeyError(f"存在しないmodel_indexがあります: {indices[positions < 0].tolist()}")

        return type(self)(
            phi_th=self.phi_th.ravel()[positions].reshape(-1, 1),
            theta0=self.theta0.ravel()[positions].reshape(-1, 1),
            eps_th=self.eps_th,
            eps_b=self.eps_b,
            molecular_weight=self.molecular_weight,
            powerlaw=self.powerlaw,
            distance=self.distance,
            model_index=self.model_indices.ravel()[positions].reshape(-1, 1),
        )

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Self:
        nrows = int(df.attrs[MetadataNames.GRID_NROWS])
        ncols = int(df.attrs[MetadataNames.GRID_NCOLS])
        if nrows <= 0 or ncols <= 0 or len(df) != nrows*ncols:
            raise ValueError("格子形状とDataFrameの行数が一致しません.")
        ordered = df.sort_values(ColumnNames.MODEL_INDEX)
        model_index = ordered[ColumnNames.MODEL_INDEX].to_numpy()
        if not pd.Index(model_index).is_unique:
            raise ValueError(
                "model_indexに重複があってはなりません."
            )

        return cls(
            phi_th=ordered[ColumnNames.PHI_TH]
                .to_numpy(dtype=np.float64).reshape(nrows, ncols),
            theta0=ordered[ColumnNames.THETA0]
                .to_numpy(dtype=np.float64).reshape(nrows, ncols),
            model_index=model_index.reshape(nrows, ncols),
            eps_th=float(df.attrs[MetadataNames.EPS_TH]),
            eps_b=float(df.attrs[MetadataNames.EPS_B]),
            molecular_weight=MolecularWeight(
                mu_e=float(df.attrs[MetadataNames.MU_E]),
                mu=float(df.attrs[MetadataNames.MU]),
            ),
            powerlaw=PowerlawParameters(
                p_index=float(df.attrs[MetadataNames.P_INDEX]),
                eps_e=float(df.attrs[MetadataNames.EPS_E]),
            ),
            distance=QuantityData.from_attrs(
                df,
                key=MetadataNames.DISTANCE_VALUE,
                unit_key=MetadataNames.DISTANCE_UNIT,
            ),
        )

    @property
    def theta(self):
        return DimensionlessElectronTemperature.hot(self.theta0)

    @property
    def a_wind(self)->u.Quantity:
        return (
            4.0*A_CYC/(9.0*self.eps_b)
            *(self.phi_th/self.theta**2)**2
        )

    @property
    def effmass_ep(self):
        return self.molecular_weight.to_effective_mass_ratio()

    @property
    def beta_sh(
        self,
    )->NDArray64:
        beta2 = (32.0*self.effmass_ep*self.theta0/(9.0*self.eps_th))
        return np.sqrt(beta2)

    @property
    def r_norm(self):
        return np.sqrt(
            32.0*self.effmass_ep*self.theta0/(9.0*self.eps_th)
        )

    @property
    def gamma_min(self):
        return np.ones_like(self.theta0)

    @property
    def doppler_delta(self):
        return quantity_converter.beta_into_doppler_delta(self.beta_sh)

    @property
    def f_factor(self):
        return MaxwellJuttnerFactor.exact(self.theta)

    @property
    def g_factor(self):
        return PowerlawDistributionFactor.exact(
            p=self.powerlaw.p_index,
            theta=self.theta,
            gamma_min=self.gamma_min,
        )

    @staticmethod
    def phi(
        t:u.Quantity,
        nu:u.Quantity
    )->NDArray64:
        return np.asarray((t*nu).decompose(),dtype=np.float64)

    def xm(
        self,
        t:u.Quantity,
        nu:u.Quantity
    )->NDArray64:
        phi = np.atleast_1d(self.phi(t, nu))
        phi_th = self.phi_th.reshape(-1)

        return phi[None, :] / phi_th[:, None]

    @staticmethod
    def ip(
        xm:NDArray64,
        table: ThermalSynchrotronIntegrals
    )->NDArray64:
        return table.ip(xm)

FOUR_PI_OVER_THREE = 4.0*np.pi/3.0

def calculate_j0(
    eps_th: float,
    eps_b: float,
    theta0: NDArray,
    theta : NDArray,
    phi_th: NDArray,
)->NDArray[np.float64]:
    return (
        FOUR_PI_OVER_THREE**3
        *(phi_th/theta**2)**3
        *eps_th/(8.0*np.pi*eps_b*theta0)
    )

def calculate_alpha0(
    eps_th: float,
    eps_b: float,
    theta0: NDArray,
    theta : NDArray,
    phi_th: NDArray,
)->NDArray:
    return (
        FOUR_PI_OVER_THREE
        *(phi_th/theta**2)
        *eps_th/(8.0*np.pi*eps_b*theta0*theta**5)
    )

@dataclass(frozen=True,slots=True)
class SynchrotronNormalizationFactor:
    norm_j: NDArray64
    norm_alpha: NDArray64

    @classmethod
    def from_grid(
        cls,
        eps_th: float,
        eps_b: float,
        theta0: NDArray64,
        theta : NDArray64,
        phi_th: NDArray64,
    )->Self:
        return cls(
            norm_j = calculate_j0(
                eps_th= eps_th,
                eps_b = eps_b,
                theta0 = theta0,
                theta = theta,
                phi_th = phi_th
            ),
            norm_alpha = calculate_alpha0(
                eps_th= eps_th,
                eps_b = eps_b,
                theta0 = theta0,
                theta = theta,
                phi_th = phi_th
            )
        )

@dataclass(frozen=True,slots=True)
class EmissivityFequencyDependenceTerm:
    thermal: NDArray64
    powerlaw: NDArray64

    @staticmethod
    def compute_thermal(
        x: NDArray,
        f_factor: NDArray,
        ip: NDArray
    )->NDArray:
        return (
            np.sqrt(3.0)*f_factor*x*ip/(8.0*np.pi)
        )

    @staticmethod
    def compute_powerlaw(
        cj: float,
        eps_e: float,
        g_factor: NDArray,
        x: NDArray,
        p:float
    )->NDArray:
        return (
            cj*eps_e*g_factor*x**(-0.5*(p-1.0))
        )

    @classmethod
    def from_grid(
        cls,
        cj: float,
        p:float,
        eps_e: float,
        x: NDArray64,
        ip: NDArray64,
        f_factor: NDArray64,
        g_factor: NDArray,
    )->Self:
        return cls(
            thermal = cls.compute_thermal(x,f_factor,ip),
            powerlaw = cls.compute_powerlaw(cj,eps_e,g_factor,x,p)
        )

    @property
    def total(self)->NDArray64:
        return self.thermal+self.powerlaw

@dataclass(frozen=True,slots=True)
class AbsorptionCoefficientFrequencyDependence:
    thermal: NDArray64
    powerlaw: NDArray64

    @staticmethod
    def compute_thermal(
        x: NDArray,
        f_factor: NDArray,
        ip: NDArray
    )->NDArray[np.float64]:
        return (
            np.pi*f_factor*ip/(3.0**1.5*x)
        )

    @staticmethod
    def compute_powerlaw(
        ca: float,
        eps_e: float,
        g_factor: NDArray,
        x: NDArray,
        p: float
    )->NDArray:
        return (
            ca*eps_e*g_factor*x**(-0.5*(p+4.0))
        )

    @classmethod
    def from_grid(
        cls,
        ca: float,
        p:float,
        eps_e: float,
        x: NDArray64,
        ip: NDArray64,
        f_factor: NDArray64,
        g_factor: NDArray,
    )->Self:
        return cls(
            thermal = cls.compute_thermal(x,f_factor,ip),
            powerlaw = cls.compute_powerlaw(ca,eps_e,g_factor,x,p)
        )

    @property
    def total(self)->NDArray64:
        return self.thermal+self.powerlaw

@dataclass(frozen=True,slots=True)
class FrequencyDependence:
    j:EmissivityFequencyDependenceTerm
    alpha:AbsorptionCoefficientFrequencyDependence

    @classmethod
    def compute_from(
        cls,
        ca: float,
        cj:float,
        p:float,
        eps_e: float,
        x: NDArray64,
        ip: NDArray64,
        f_factor: NDArray64,
        g_factor: NDArray,
    )->Self:
        return cls(
            j=EmissivityFequencyDependenceTerm.from_grid(
                cj=cj,
                p=p,
                eps_e=eps_e,
                x=x,
                ip=ip,
                f_factor=f_factor,
                g_factor=g_factor
            ),
            alpha=AbsorptionCoefficientFrequencyDependence.from_grid(
                ca=ca,
                p=p,
                eps_e=eps_e,
                x=x,
                ip=ip,
                f_factor=f_factor,
                g_factor=g_factor
            )
        )

@dataclass(frozen=True,slots=True)
class Emissivity:
    norm: NDArray[np.float64]
    zeta: EmissivityFequencyDependenceTerm

@dataclass(frozen=True,slots=True)
class AbsorptionCoefficient:
    norm: NDArray[np.float64]
    eta: AbsorptionCoefficientFrequencyDependence

@dataclass(frozen=True,slots=True)
class SynchrotronComponents:
    thermal_only: NDArray64
    powerlaw_only: NDArray64
    total: NDArray64

@dataclass(frozen=True,slots=True)
class SynchrotronQuantities:
    thermal_only: u.Quantity
    powerlaw_only: u.Quantity
    total: u.Quantity

@dataclass(frozen=True,slots=True)
class SourceFunction(SynchrotronComponents):

    @classmethod
    def compute_from(
        cls,
        jnu: Emissivity,
        anu: AbsorptionCoefficient
    )->Self:
        norm_s = jnu.norm/anu.norm
        zeta = jnu.zeta
        eta = anu.eta
        return cls(
            thermal_only = norm_s*zeta.thermal/eta.thermal,
            powerlaw_only = norm_s*zeta.powerlaw/eta.powerlaw,
            total = norm_s*(zeta.total)/(eta.total)
        )

@dataclass(frozen=True,slots=True)
class OpticalDepth(SynchrotronComponents):

    @classmethod
    def compute_from(
        cls,
        anu: AbsorptionCoefficient,
        norm_r: NDArray64
    )->Self:
        norm_a = anu.norm
        eta = anu.eta
        norm_tau = norm_a*norm_r
        return cls(
            thermal_only = norm_tau*eta.thermal,
            powerlaw_only = norm_tau*eta.powerlaw,
            total = norm_tau*eta.total,
        )

@dataclass(frozen=True,slots=True)
class SpecificLuminosity(SynchrotronComponents):

    @classmethod
    def compute_from(
        cls,
        norm_r: NDArray64,
        snu: SourceFunction,
        tau: OpticalDepth
    )->Self:
        norm_l = (2.0*np.pi*norm_r)**2
        return cls(
            thermal_only = norm_l*snu.thermal_only*(-np.expm1(-tau.thermal_only)),
            powerlaw_only = norm_l*snu.powerlaw_only*(-np.expm1(-tau.powerlaw_only)),
            total = norm_l*snu.total*(-np.expm1(-tau.total)),
        )

    @staticmethod
    def as_quantity(lambda_syn:NDArray64)->u.Quantity:
        return u.Quantity(lambda_syn*mec2)

    @property
    def thermal_quantity(self):
        return self.as_quantity(self.thermal_only)

    @property
    def powerlaw_quantity(self):
        return self.as_quantity(self.powerlaw_only)

    @property
    def total_quantity(self):
        return self.as_quantity(self.total)

@dataclass(frozen=True,slots=True)
class ObserverframeSpecificFlux(SynchrotronQuantities):
    pass

@dataclass(frozen=True,slots=True)
class SourceframeSpecificFlux(SynchrotronQuantities):

    @staticmethod
    def convert(
        lnu: u.Quantity,
        distance: u.Quantity
    )->u.Quantity:
        return quantity_converter.lnu_into_fnu(lnu,distance)

    @classmethod
    def from_luminosity(
        cls,
        lnu: SpecificLuminosity,
        distance: QuantityData
    )->Self:
        d_quantity = distance.quantity
        return cls(
            thermal_only = cls.convert(lnu.thermal_quantity,d_quantity),
            powerlaw_only = cls.convert(lnu.powerlaw_quantity,d_quantity),
            total = cls.convert(lnu.total_quantity,d_quantity)
        )

    @staticmethod
    def convert_into_obsframe(
        fnu_srcframe:u.Quantity,
        doppler_factor:NDArray64
    )->u.Quantity:
        doppler_factor_column = u.Quantity(doppler_factor[:,np.newaxis])
        return doppler_factor_column**3 * fnu_srcframe

    def to_observer_frame(
        self,
        doppler_factor:NDArray64
    )->ObserverframeSpecificFlux:
        return ObserverframeSpecificFlux(
            thermal_only=self.convert_into_obsframe(self.thermal_only,doppler_factor),
            powerlaw_only=self.convert_into_obsframe(self.powerlaw_only,doppler_factor),
            total=self.convert_into_obsframe(self.total,doppler_factor),
        )

@dataclass(frozen=True,slots=True)
class SynchrotronPeak:
    column_index: NDArray[np.intp]
    xm: NDArray64
    phi: u.Quantity
    fnu_total: u.Quantity
    lnu: u.Quantity
    model_index: NDArray[np.int64] | None = None

    def __post_init__(self) -> None:
        model_index = (
            np.arange(self.xm.shape[0], dtype=np.int64).reshape(-1, 1)
            if self.model_index is None else np.asarray(self.model_index)
        )
        if model_index.shape != self.xm.shape or model_index.dtype.kind not in "iu":
            raise ValueError("model_indexはpeakと同じ形状の整数配列が必要です.")
        if not pd.Index(model_index.ravel()).is_unique:
            raise ValueError("model_indexに重複があってはなりません.")
        object.__setattr__(self, "model_index", model_index.astype(np.int64))

    @property
    def model_indices(self) -> NDArray[np.int64]:
        if self.model_index is None:
            raise RuntimeError("model_indexが初期化されていません.")
        return self.model_index

    @property
    def phi_grid(self) -> QuantityGrid:
        df = self.to_df()
        return QuantityGrid.from_df(
            df,
            column=PhiName.PHI,
            unit=df.attrs[MetadataNames.PHI_UNIT],
            index_name=ColumnNames.MODEL_INDEX,
        )

    @property
    def fnu_grid(self) -> QuantityGrid:
        df = self.to_df()
        return QuantityGrid.from_df(
            df,
            column=FluxName.FNU_TOTAL,
            unit=df.attrs[MetadataNames.FNU_UNIT],
            index_name=ColumnNames.MODEL_INDEX,
        )

    @property
    def index(self) -> IndexGrid:
        df = self.to_df()
        ordered = df.sort_values(ColumnNames.MODEL_INDEX)
        return IndexGrid.from_df(ordered.set_index(ColumnNames.MODEL_INDEX))

    def _parameter_grid(
        self,
        model_params: ModelParameters,
        values: u.Quantity,
    ) -> QuantityGrid:
        index_grid = self.index
        indices = index_grid.values.ravel()
        positions = pd.Index(model_params.model_indices.ravel()).get_indexer(indices)
        if np.any(positions < 0):
            raise KeyError(f"対応するmodel_indexがありません: {indices[positions < 0].tolist()}")
        quantity = values.reshape(-1)[positions].reshape(index_grid.shape)
        return QuantityGrid(
            values=np.asarray(quantity.value, dtype=np.float64),
            unit=quantity.unit.to_string(),
        )

    def a_wind_grid(self, model_params: ModelParameters) -> QuantityGrid:
        return self._parameter_grid(model_params, model_params.a_wind)

    def beta_sh_grid(self, model_params: ModelParameters) -> QuantityGrid:
        return self._parameter_grid(model_params, u.Quantity(model_params.beta_sh))

    def theta_grid(self, model_params: ModelParameters) -> QuantityGrid:
        return self._parameter_grid(model_params, u.Quantity(model_params.theta))

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Self:
        ordered = df.sort_values(ColumnNames.MODEL_INDEX)
        model_index = ordered[ColumnNames.MODEL_INDEX].to_numpy()
        if not pd.Index(model_index).is_unique:
            raise ValueError(
                "model_indexに重複があってはなりません."
            )

        fnu_unit = df.attrs[MetadataNames.FNU_UNIT]
        phi_unit = df.attrs[MetadataNames.PHI_UNIT]
        lnu_unit = df.attrs[MetadataNames.LNU_UNIT]
        return cls(
            model_index=model_index.reshape(-1, 1),
            column_index=ordered[ColumnNames.PEAK_COLUMN_INDEX]
                .to_numpy(dtype=np.intp).reshape(-1, 1),
            xm=ordered[ColumnNames.XM]
                .to_numpy(dtype=np.float64).reshape(-1, 1),
            phi=u.Quantity(
                ordered[PhiName.PHI].to_numpy(dtype=np.float64).reshape(-1, 1),
                unit=phi_unit,
            ).to(u.dimensionless_unscaled),
            fnu_total=u.Quantity(
                ordered[FluxName.FNU_TOTAL]
                    .to_numpy(dtype=np.float64).reshape(-1, 1),
                unit=fnu_unit,
            ),
            lnu=u.Quantity(
                ordered[ColumnNames.LNU]
                    .to_numpy(dtype=np.float64).reshape(-1, 1),
                unit=lnu_unit,
            ),
        )

    def to_df(
        self,
        *,
        fnu_unit: str = DefaultUnits.fnu,
        phi_unit: str = DefaultUnits.phi,
        lnu_unit: str = DefaultUnits.lnu,
    ) -> pd.DataFrame:
        df = pd.DataFrame({
            ColumnNames.MODEL_INDEX: self.model_indices.ravel(),
            ColumnNames.PEAK_COLUMN_INDEX: self.column_index.ravel(),
            ColumnNames.XM: self.xm.ravel(),
            PhiName.PHI: self.phi.to_value(phi_unit).ravel(),
            FluxName.FNU_TOTAL: self.fnu_total.to_value(fnu_unit).ravel(),
            ColumnNames.LNU: self.lnu.to_value(lnu_unit).ravel(),
        })
        df.attrs[MetadataNames.FNU_UNIT] = str(fnu_unit)
        df.attrs[MetadataNames.PHI_UNIT] = str(phi_unit)
        df.attrs[MetadataNames.LNU_UNIT] = str(lnu_unit)
        return df

@dataclass(frozen=True,slots=True)
class SynchrotronResults:
    model_params:ModelParameters
    xm: NDArray64
    j_nu: Emissivity
    alpha_nu: AbsorptionCoefficient
    tau: OpticalDepth
    lnu: SpecificLuminosity
    fnu_obsframe: ObserverframeSpecificFlux

    @property
    def phi(self) -> u.Quantity:
        return u.Quantity(
            self.xm*self.model_params.phi_th.reshape(-1,1),
            u.dimensionless_unscaled,
        )

    def to_lightcurve(self, nu: QuantityData) -> "SynchrotronLightcurves":
        return SynchrotronLightcurves(
            model_params=self.model_params,
            xm=self.xm,
            j_nu=self.j_nu,
            alpha_nu=self.alpha_nu,
            tau=self.tau,
            lnu=self.lnu,
            fnu_obsframe=self.fnu_obsframe,
            nu=nu,
        )

    @property
    def peak(self) -> SynchrotronPeak:
        fnu_total = self.fnu_obsframe.total
        column_index = np.argmax(fnu_total.value, axis=1)[:, np.newaxis]
        row_index = np.arange(self.xm.shape[0])[:, np.newaxis]
        fnu_total_peak = fnu_total[row_index, column_index]
        if not isinstance(fnu_total_peak, u.Quantity):
            raise TypeError(
                "fnu_total_peak をQuantityとして返せません."
            )
        phi_peak = self.phi[row_index, column_index]
        if not isinstance(phi_peak, u.Quantity):
            raise TypeError(
                "phi_peak をQuantityとして返せません."
            )
        lnu_peak = self.lnu.total_quantity[row_index, column_index]
        if not isinstance(lnu_peak, u.Quantity):
            raise TypeError(
                "lnu_peak をQuantityとして返せません."
            )

        return SynchrotronPeak(
            column_index=column_index,
            xm=self.xm[row_index, column_index],
            phi=phi_peak,
            fnu_total=fnu_total_peak,
            lnu=lnu_peak,
            model_index=self.model_params.model_indices.reshape(-1, 1),
        )

    def to_df(
        self,
        *,
        fnu_unit: str = DefaultUnits.fnu,
        phi_unit: str = DefaultUnits.phi,
    ) -> pd.DataFrame:
        _, n_xm = self.xm.shape
        fnu = self.fnu_obsframe

        df = pd.DataFrame({
            ColumnNames.MODEL_INDEX: np.repeat(self.model_params.model_indices.ravel(), n_xm),
            ColumnNames.XM: self.xm.ravel(),
            PhiName.PHI: self.phi.to_value(phi_unit).ravel(),
            FluxName.FNU_THERMAL: fnu.thermal_only.to_value(fnu_unit).ravel(),
            FluxName.FNU_POWERLAW: fnu.powerlaw_only.to_value(fnu_unit).ravel(),
            FluxName.FNU_TOTAL: fnu.total.to_value(fnu_unit).ravel(),
        })
        df.attrs[MetadataNames.FNU_UNIT] = str(fnu_unit)
        df.attrs[MetadataNames.PHI_UNIT] = str(phi_unit)
        return df

    @classmethod
    def compute_from(
        cls,
        model_params:ModelParameters,
        xm:NDArray64,
        table: ThermalSynchrotronIntegrals,
    )->Self:
        # 格子の各セルを一つのモデルとして扱う。
        n_models = model_params.phi_th.size
        if model_params.theta0.shape != model_params.phi_th.shape:
            raise ValueError("phi_thとtheta0の格子形状が一致しません")
        if xm.ndim != 2 or xm.shape[0] != 1:
            raise ValueError(
                f"xmは(1, M)の配列が必要です: shape={xm.shape}"
            )

        # xmを全モデルへ広げ、モデル格子を(n_models, 1)にする。
        xm_grid = np.broadcast_to(xm, (n_models, xm.shape[1]))
        r = model_params.r_norm.reshape(-1, 1)
        norms = SynchrotronNormalizationFactor.from_grid(
            eps_th=model_params.eps_th,
            eps_b=model_params.eps_b,
            theta0=model_params.theta0,
            theta=model_params.theta,
            phi_th=model_params.phi_th,
        )
        x_dependent_terms = FrequencyDependence.compute_from(
            ca=model_params.powerlaw.ca,
            cj=model_params.powerlaw.cj,
            p=model_params.powerlaw.p_index,
            eps_e=model_params.powerlaw.eps_e,
            x=xm_grid,
            ip=model_params.ip(xm,table),
            f_factor=model_params.f_factor.reshape(-1,1),
            g_factor=model_params.g_factor.reshape(-1,1),
        )
        jnu = Emissivity(
            norm=norms.norm_j.reshape(-1, 1),
            zeta=x_dependent_terms.j
        )
        alpha_nu = AbsorptionCoefficient(
            norm=norms.norm_alpha.reshape(-1, 1),
            eta=x_dependent_terms.alpha
        )
        snu = SourceFunction.compute_from(jnu,alpha_nu)
        tau = OpticalDepth.compute_from(alpha_nu,r)
        lnu = SpecificLuminosity.compute_from(r,snu,tau)
        fnu_srcframe = SourceframeSpecificFlux.from_luminosity(lnu,model_params.distance)
        fnu_obsframe = fnu_srcframe.to_observer_frame(
            np.asarray(model_params.doppler_delta).reshape(-1)
        )

        return cls(
            model_params=model_params,
            xm=xm_grid,
            j_nu=jnu,
            alpha_nu=alpha_nu,
            tau=tau,
            lnu=lnu,
            fnu_obsframe=fnu_obsframe
        )

@dataclass(frozen=True, slots=True)
class SynchrotronLightcurves(SynchrotronResults):
    nu: QuantityData

    @property
    def t(self) -> u.Quantity:
        return self.phi / self.nu.quantity

    def to_df(
        self,
        *,
        fnu_unit: str = DefaultUnits.fnu,
        phi_unit: str = DefaultUnits.phi,
        nu_unit: str = DefaultUnits.nu,
        t_unit: str = DefaultUnits.t,
    ) -> pd.DataFrame:
        df = SynchrotronResults.to_df(
            self,
            fnu_unit=fnu_unit,
            phi_unit=phi_unit,
        )
        df[ColumnNames.NU] = self.nu.value_in(nu_unit)
        df[ColumnNames.T] = self.t.to_value(t_unit).ravel()
        df.attrs[MetadataNames.NU_UNIT] = str(nu_unit)
        df.attrs[MetadataNames.T_UNIT] = str(t_unit)
        return df

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

