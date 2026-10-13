# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false

from typing import Hashable, Literal, Self
import astropy.units as u
from matplotlib.axes import Axes
from module import quantity_converter, utils
from module.electron_distributions import(
    MaxwellJuttnerFactor,
    PowerlawNormalization,
    MaxwellJuttnerDistribution
)
import numpy as np
import pandas as pd
from enum import StrEnum, auto
from dataclasses import dataclass
from module.mydataclasses import (
    FloatVector, FloatVectorSpace,
    QuantityArrayBase, QuantityData,
    QuantityGrid, IndexGrid,
)
from module.synchrotron_numerical_table import ThermalSynchrotronIntegrals
from module.types import FloatArray
from module.synchrotron_coefficients import (
    SynchrotronNormalizationFactor,
    EmissivityFequencyDependenceTerm,
    AbsorptionCoefficientFrequencyDependence,
    FrequencyDependence,
    Emissivity,
    AbsorptionCoefficient,
    FOUR_PI_OVER_THREE,
    POWERLAW_FLOOR,
)
from module.thermal_synchrotron import DimensionlessThermalLuminosity
from scipy import special
from numpy.typing import NDArray
from astropy.constants import e,c,m_e,m_p
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
SYNCHROTRON_MODEL = Literal["thermal","total"]

class DefaultUnits(StrEnum):
    fnu = "mJy"
    lnu = "erg s-1 Hz-1"
    t = "min"
    nu = "GHz"
    phi = "GHz min"

class DimensionlessSynchrotronKeys(StrEnum):
    ZETA_THERMAL = auto()
    ZETA_POWERLAW = auto()
    ZETA_TOTAL = auto()
    ETA_THERMAL = auto()
    ETA_POWERLAW = auto()
    ETA_TOTAL = auto()
    SNU_THERMAL = auto()
    SNU_POWERLAW = auto()
    SNU_TOTAL = auto()
    LAMBDA_THERMAL = auto()
    LAMBDA_POWERLAW = auto()
    LAMBDA_TOTAL = auto()

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
    F_E = auto()
    MU_E_MU = auto()
    DISTANCE_VALUE = auto()
    DISTANCE_UNIT = auto()
    A_WIND_UNIT = auto()
    PHI_UNIT = auto()
    LNU_UNIT = auto()
    FNU_UNIT = auto()
    PEAK_COMPONENT = auto()
    NU_UNIT = auto()
    T_UNIT = auto()
    BIN_WIDTH = auto()
    SYNCHROTRON_MODEL = auto()
    THERMAL_MODEL_INDEX = auto()

class FluxName(StrEnum):
    FNU_THERMAL = auto()
    FNU_POWERLAW = auto()
    FNU_TOTAL = auto()
    FNU_WITH_BG= auto()
    FNU_NET = auto()

class PhiName(StrEnum):
    PHI = auto()

class SwitchKeys(StrEnum):
    THERMAL_ONLY = auto()
    EXACT_DIST_FACTORS = auto()

class ColumnNames(StrEnum):
    THERMAL_MODEL_INDEX = auto()
    POWERLAW_SUBMODEL_INDEX = auto()
    CANDIDATE_MODEL_INDEX = auto()
    MODEL_INDEX = auto()
    PEAK_COLUMN_INDEX = auto()
    XM = auto()
    PHI_TH = auto()
    THETA0 = auto()
    PHI_TH_THIN = auto()
    PHI_PEAK = auto()
    PHI_PEAK_ERR = auto()
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
    FNU_PEAK_NET = auto()
    LNU_PEAK_THIN = auto()
    NU = auto()
    T = auto()
    T_CENTER = auto()
    T_LEFT = auto()
    T_RIGHT = auto()
    T_PEAK = auto()
    FNU_OBS_FRAME = auto()
    LNU = auto()
    FNU_SRC_FRAME = auto()
    FNU_FLARE = auto()
    FNU_TOTAL = auto()
    FNU_BG = auto()
    JNU_THERMAL = auto()

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
    mu_e: float
    mu: float

    def to_ratio(
        self
    )->float:
        return float(
            (self.mu_e/self.mu)
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
    f_e: float

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
class CommonShockParameters:
    eps_th: float
    eps_b: float
    molecular_weight: MolecularWeight

@dataclass(frozen=True,slots=True)
class ShockParameters:
    common: CommonShockParameters
    powerlaw: PowerlawParameters

@dataclass(frozen=True,slots=True)
class InputParametersRaw:
    phi_th: QuantityArrayBase
    theta0: FloatVectorSpace
    distance: QuantityData
    shock: ShockParameters
    switches: dict[str,bool]

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
    switches:dict[str,bool]
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
            eps_th=input.shock.common.eps_th,
            eps_b =input.shock.common.eps_b,
            molecular_weight=input.shock.common.molecular_weight,
            distance=input.distance,
            powerlaw=input.shock.powerlaw,
            switches=input.switches
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
            MetadataNames.F_E: float(self.powerlaw.f_e),
            MetadataNames.DISTANCE_VALUE: float(self.distance.value),
            MetadataNames.DISTANCE_UNIT: self.distance.unit,
            **self.switches,
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
            switches=self.switches,
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

        switch_keys: set[str] = {switch.value for switch in SwitchKeys}

        switches: dict[str, bool] = {
            str(key): utils.str_to_bool(value)
            for key, value in df.attrs.items()
            if isinstance(key, str) and key in switch_keys
        }

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
                f_e=float(df.attrs[MetadataNames.F_E]),
            ),
            switches=switches,
            distance=QuantityData.from_attrs(
                df,
                key=MetadataNames.DISTANCE_VALUE,
                unit_key=MetadataNames.DISTANCE_UNIT,
            ),
        )

    @property
    def theta(self):
        return DimensionlessElectronTemperature.gp98(self.theta0)

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
        """
            minimum Lorentz factor for electrons to be acceralated
            Margalit&Quataert(2021)
        """
        return MaxwellJuttnerDistribution.calculate_mean_gamma(self.theta)

    @property
    def xm_cutoff(self) -> NDArray64:
        """Return (gamma_min / theta)^2 with the model grid shape"""
        return (self.gamma_min / self.theta)**2

    @property
    def doppler_delta(self):
        return quantity_converter.beta_into_doppler_delta(self.beta_sh)

    @property
    def f_factor(self):
        use_exact = self.switches[SwitchKeys.EXACT_DIST_FACTORS]
        f_factor = MaxwellJuttnerFactor.compute(self.theta)
        if use_exact:
            return f_factor.exact
        else:
            return f_factor.hot

    @property
    def g_factor(self)->NDArray[np.float64]:
        use_exact = self.switches[SwitchKeys.EXACT_DIST_FACTORS]
        g_factor = PowerlawNormalization.compute(
            p=self.powerlaw.p_index,
            theta=self.theta,
            gamma_min=self.gamma_min
        )
        if use_exact:
            return g_factor.exact
        else:
            return g_factor.hot

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

    @property
    def log10_tau0(
        self
    )->NDArray[np.float64]:
        log10_theta0    = np.log10(self.theta0)
        log10_theta     = np.log10(self.theta)
        log10_epsb      = np.log10(self.eps_b)
        log10_epsth     = np.log10(self.eps_th)
        log10_phith     = np.log10(self.phi_th)
        log10_mep       = np.log10(self.molecular_weight.to_effective_mass_ratio())

        return(
            -7.0*log10_theta
            -0.5*(log10_theta0-log10_epsth)
            +1.5*np.log10(2.0)
            -2.0*np.log10(3.0)
            -log10_epsb
            +log10_phith
            +0.5*log10_mep
        )

    @property
    def log10_tauth(self)->NDArray[np.float64]:
        return (
            np.log10(np.pi)
            -1.5*np.log10(3.0)
            +np.log10(self.f_factor)
            +self.log10_tau0
        )


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
class SourceFunctionFrequencyDependence:
    thermal: NDArray64
    powerlaw: NDArray64
    total: NDArray64

    @classmethod
    def compute_from(
        cls,
        j_scr: EmissivityFequencyDependenceTerm,
        alpha_scr: AbsorptionCoefficientFrequencyDependence,
    )->Self:
        return cls(
            thermal=j_scr.thermal/alpha_scr.thermal,
            powerlaw=j_scr.powerlaw/alpha_scr.powerlaw,
            total=j_scr.total/alpha_scr.total,
        )

@dataclass(frozen=True,slots=True)
class SourceFunction:
    norm: NDArray64
    dependence: SourceFunctionFrequencyDependence

    @classmethod
    def compute_from(
        cls,
        jnu: Emissivity,
        anu: AbsorptionCoefficient
    )->Self:
        return cls(
            norm=jnu.norm/anu.norm,
            dependence=SourceFunctionFrequencyDependence.compute_from(
                j_scr=jnu.zeta,
                alpha_scr=anu.eta,
            ),
        )

    @property
    def thermal_only(self)->NDArray64:
        return self.norm*self.dependence.thermal

    @property
    def powerlaw_only(self)->NDArray64:
        return self.norm*self.dependence.powerlaw

    @property
    def total(self)->NDArray64:
        return self.norm*self.dependence.total

@dataclass(frozen=True,slots=True)
class OpticalDepth(SynchrotronComponents):

    @classmethod
    def compute_from(
        cls,
        anu: AbsorptionCoefficient,
        norm_r: NDArray64
    )->Self:
        alpha0 = anu.norm
        eta = anu.eta
        tau0 = alpha0*norm_r
        return cls(
            thermal_only = tau0*eta.thermal,
            powerlaw_only = tau0*eta.powerlaw,
            total = tau0*eta.total,
        )

@dataclass(frozen=True,slots=True)
class SpecificLuminosityFrequencyDependence:
    thermal: NDArray64
    powerlaw: NDArray64
    total: NDArray64

    @classmethod
    def compute_from(
        cls,
        sigma: SourceFunctionFrequencyDependence,
        tau: OpticalDepth
    )->Self:
        return cls(
            thermal=sigma.thermal*(-np.expm1(-tau.thermal_only)),
            powerlaw=sigma.powerlaw*(-np.expm1(-tau.powerlaw_only)),
            total=sigma.total*(-np.expm1(-tau.total)),
        )


@dataclass(frozen=True,slots=True)
class SpecificLuminosity:
    norm: NDArray64
    dependence: SpecificLuminosityFrequencyDependence

    @classmethod
    def compute_from(
        cls,
        norm_r: NDArray64,
        snu: SourceFunction,
        tau: OpticalDepth
    )->Self:
        return cls(
            norm=(2.0*np.pi*norm_r)**2*snu.norm,
            dependence=SpecificLuminosityFrequencyDependence.compute_from(
                sigma=snu.dependence,
                tau=tau,
            ),
        )

    @property
    def thermal_only(self)->NDArray64:
        return self.norm*self.dependence.thermal

    @property
    def powerlaw_only(self)->NDArray64:
        return self.norm*self.dependence.powerlaw

    @property
    def total(self)->NDArray64:
        return self.norm*self.dependence.total

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
    fnu: u.Quantity
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
            column=ColumnNames.FNU_PEAK,
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
        fnu_column = (
            ColumnNames.FNU_PEAK
            if ColumnNames.FNU_PEAK in ordered.columns
            else FluxName.FNU_TOTAL
        )
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
            fnu=u.Quantity(
                ordered[fnu_column]
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
            ColumnNames.FNU_PEAK: self.fnu.to_value(fnu_unit).ravel(),
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
    snu: SourceFunction
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
            snu=self.snu,
            tau=self.tau,
            lnu=self.lnu,
            fnu_obsframe=self.fnu_obsframe,
            nu=nu,
        )

    def peak(
        self,
    ) -> SynchrotronPeak:
        thermal_only = self.model_params.switches[SwitchKeys.THERMAL_ONLY]
        if thermal_only:
            fnu = self.fnu_obsframe.thermal_only
            lnu = self.lnu.thermal_quantity
        else :
            fnu = self.fnu_obsframe.total
            lnu = self.lnu.total_quantity

        column_index = np.argmax(fnu.value, axis=1)[:, np.newaxis]
        row_index = np.arange(self.xm.shape[0])[:, np.newaxis]
        fnu_peak = fnu[row_index, column_index]
        if not isinstance(fnu_peak, u.Quantity):
            raise TypeError(
                "fnu_peak をQuantityとして返せません."
            )
        phi_peak = self.phi[row_index, column_index]
        if not isinstance(phi_peak, u.Quantity):
            raise TypeError(
                "phi_peak をQuantityとして返せません."
            )
        lnu_peak = lnu[row_index, column_index]
        if not isinstance(lnu_peak, u.Quantity):
            raise TypeError(
                "lnu_peak をQuantityとして返せません."
            )

        return SynchrotronPeak(
            column_index=column_index,
            xm=self.xm[row_index, column_index],
            phi=phi_peak,
            fnu=fnu_peak,
            lnu=lnu_peak,
            model_index=self.model_params.model_indices.reshape(-1, 1),
        )

    def to_df(
        self,
        *,
        fnu_unit: str = DefaultUnits.fnu,
        phi_unit: str = DefaultUnits.phi,
        index_column_name: str = ColumnNames.MODEL_INDEX
    ) -> pd.DataFrame:
        _, n_xm = self.xm.shape
        fnu = self.fnu_obsframe
        jnu = self.j_nu

        dataset = {
            ColumnNames.MODEL_INDEX: np.repeat(self.model_params.model_indices.ravel(), n_xm),
            ColumnNames.XM: self.xm.ravel(),
            PhiName.PHI: self.phi.to_value(phi_unit).ravel(),
            FluxName.FNU_THERMAL: fnu.thermal_only.to_value(fnu_unit).ravel(),
        }

        df = pd.DataFrame({
            index_column_name: np.repeat(self.model_params.model_indices.ravel(), n_xm),
            ColumnNames.XM: self.xm.ravel(),
            PhiName.PHI: self.phi.to_value(phi_unit).ravel(),
            FluxName.FNU_THERMAL: fnu.thermal_only.to_value(fnu_unit).ravel(),
            FluxName.FNU_POWERLAW: fnu.powerlaw_only.to_value(fnu_unit).ravel(),
            FluxName.FNU_TOTAL: fnu.total.to_value(fnu_unit).ravel(),
            DimensionlessSynchrotronKeys.ZETA_THERMAL: jnu.zeta.thermal.ravel(),
            DimensionlessSynchrotronKeys.ZETA_POWERLAW: jnu.zeta.powerlaw.ravel(),
            DimensionlessSynchrotronKeys.ZETA_TOTAL: jnu.zeta.total.ravel(),
            DimensionlessSynchrotronKeys.ETA_THERMAL: self.alpha_nu.eta.thermal.ravel(),
            DimensionlessSynchrotronKeys.ETA_POWERLAW: self.alpha_nu.eta.powerlaw.ravel(),
            DimensionlessSynchrotronKeys.ETA_TOTAL: self.alpha_nu.eta.total.ravel(),
            DimensionlessSynchrotronKeys.SNU_THERMAL: self.snu.dependence.thermal.ravel(),
            DimensionlessSynchrotronKeys.SNU_POWERLAW: self.snu.dependence.powerlaw.ravel(),
            DimensionlessSynchrotronKeys.SNU_TOTAL: self.snu.dependence.total.ravel(),
            DimensionlessSynchrotronKeys.LAMBDA_THERMAL: self.lnu.dependence.thermal.ravel(),
            DimensionlessSynchrotronKeys.LAMBDA_POWERLAW: self.lnu.dependence.powerlaw.ravel(),
            DimensionlessSynchrotronKeys.LAMBDA_TOTAL: self.lnu.dependence.total.ravel(),
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
        """Compute spectra using the cutoff derived from each model."""
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
            eps_e=model_params.powerlaw.f_e,
            x=xm_grid,
            ip=model_params.ip(xm,table),
            f_factor=model_params.f_factor.reshape(-1,1),
            g_factor=model_params.g_factor.reshape(-1,1),
            xm_cutoff=model_params.xm_cutoff.ravel(),
        )
        jnu = Emissivity(
            norm=norms.j0.reshape(-1, 1),
            zeta=x_dependent_terms.j
        )
        alpha_nu = AbsorptionCoefficient(
            norm=norms.alpha0.reshape(-1, 1),
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
            snu=snu,
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
        index_column_name: str = ColumnNames.MODEL_INDEX,
    ) -> pd.DataFrame:
        df = SynchrotronResults.to_df(
            self,
            fnu_unit=fnu_unit,
            phi_unit=phi_unit,
            index_column_name=index_column_name
        )
        df[ColumnNames.NU] = self.nu.value_in(nu_unit)
        df[ColumnNames.T] = self.t.to_value(t_unit).ravel()
        df.attrs[MetadataNames.NU_UNIT] = str(nu_unit)
        df.attrs[MetadataNames.T_UNIT] = str(t_unit)
        return df
