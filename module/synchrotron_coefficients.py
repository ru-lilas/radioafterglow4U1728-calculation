# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false
"""Synchrotron emissivity and absorption coefficients and their normalization."""

from dataclasses import dataclass
from typing import Self
import astropy.units as u
from astropy.constants import e,c,m_e,m_p
c = u.Quantity(c)
m_e = u.Quantity(m_e)
m_p = u.Quantity(m_p)
mec2 = (m_e*c**2)

import numpy as np
from numpy.typing import NDArray

type NDArray64 = NDArray[np.float64]

POWERLAW_FLOOR:float = 1.0e-100

FOUR_PI_OVER_THREE = 4.0*np.pi/3.0

@dataclass(frozen=True,slots=True)
class SynchrotronNormalizationFactor:
    j0: NDArray64
    alpha0: NDArray64

    @staticmethod
    def calculate_j0(
        eps_th: float,
        eps_b: float,
        theta0: NDArray,
        theta : NDArray,
        phi_th: NDArray,
    )->NDArray[np.float64]:
        """
            devided by m_e c^2 / (ct)^3
        """
        return (
            FOUR_PI_OVER_THREE**3
            *(phi_th/theta**2)**3
            *eps_th/(8.0*np.pi*eps_b*theta0)
        )

    @staticmethod
    def calculate_alpha0(
        eps_th: float,
        eps_b: float,
        theta0: NDArray,
        theta : NDArray,
        phi_th: NDArray,
    )->NDArray:
        """
            devided by 1 / (ct)
        """
        return (
            FOUR_PI_OVER_THREE
            *(phi_th/theta**2)
            *eps_th/(8.0*np.pi*eps_b*theta0*theta**5)
        )

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
            j0 = cls.calculate_j0(
                eps_th= eps_th,
                eps_b = eps_b,
                theta0 = theta0,
                theta = theta,
                phi_th = phi_th
            ),
            alpha0 = cls.calculate_alpha0(
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
        ip: NDArray,
        f_e: float
    )->NDArray:
        return (
            (1.0-f_e)*np.sqrt(3.0)*f_factor*x*ip/(8.0*np.pi)
        )

    @staticmethod
    def compute_powerlaw(
        cj: float,
        f_e: float,
        g_factor: NDArray,
        x: NDArray,
        p:float,
        xm_cutoff: NDArray64,
    )->NDArray[np.float64]:
        if x.ndim != 2:
            raise ValueError("xは2次元配列が必要です")
        if xm_cutoff.ndim != 1 or xm_cutoff.size != x.shape[0]:
            raise ValueError("xm_cutoffはxの行数と同じ長さの1次元配列が必要です")
        jpl = cj*f_e*g_factor*x**(-0.5*(p-1.0))
        return np.where(
            x < xm_cutoff[:, None],
            POWERLAW_FLOOR,
            jpl
        )

    @classmethod
    def from_grid(
        cls,
        cj: float,
        p:float,
        f_e: float,
        x: NDArray64,
        ip: NDArray64,
        f_factor: NDArray64,
        g_factor: NDArray,
        xm_cutoff: NDArray64,
    )->Self:
        return cls(
            thermal = cls.compute_thermal(x,f_factor,ip,f_e),
            powerlaw = cls.compute_powerlaw(cj,f_e,g_factor,x,p,xm_cutoff)
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
        ip: NDArray,
        f_e: float
    )->NDArray[np.float64]:
        return (
            (1.0-f_e)*np.pi*f_factor*ip/(3.0**1.5*x)
        )

    @staticmethod
    def compute_powerlaw(
        ca: float,
        f_e: float,
        g_factor: NDArray,
        x: NDArray,
        p: float,
        xm_cutoff: NDArray64,
    )->NDArray[np.float64]:
        if x.ndim != 2:
            raise ValueError("xは2次元配列が必要です")
        if xm_cutoff.ndim != 1 or xm_cutoff.size != x.shape[0]:
            raise ValueError("xm_cutoffはxの行数と同じ長さの1次元配列が必要です")
        apl = ca*f_e*g_factor*x**(-0.5*(p+4.0))
        return np.where(
            x < xm_cutoff[:, None],
            POWERLAW_FLOOR,
            apl
        )

    @classmethod
    def from_grid(
        cls,
        ca: float,
        p:float,
        f_e: float,
        x: NDArray64,
        ip: NDArray64,
        f_factor: NDArray64,
        g_factor: NDArray,
        xm_cutoff: NDArray64,
    )->Self:
        return cls(
            thermal = cls.compute_thermal(x,f_factor,ip,f_e),
            powerlaw = cls.compute_powerlaw(ca,f_e,g_factor,x,p,xm_cutoff)
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
        f_e: float,
        x: NDArray64,
        ip: NDArray64,
        f_factor: NDArray64,
        g_factor: NDArray,
        xm_cutoff: NDArray64,
    )->Self:
        return cls(
            j=EmissivityFequencyDependenceTerm.from_grid(
                cj=cj,
                p=p,
                f_e=f_e,
                x=x,
                ip=ip,
                f_factor=f_factor,
                g_factor=g_factor,
                xm_cutoff=xm_cutoff,
            ),
            alpha=AbsorptionCoefficientFrequencyDependence.from_grid(
                ca=ca,
                p=p,
                f_e=f_e,
                x=x,
                ip=ip,
                f_factor=f_factor,
                g_factor=g_factor,
                xm_cutoff=xm_cutoff,
            )
        )

@dataclass(frozen=True,slots=True)
class Emissivity:
    j_0: NDArray[np.float64]
    j_scr: EmissivityFequencyDependenceTerm

    def j0_quantity(
        self,
        t:u.Quantity
    ):
        return u.Quantity(self.j_0*mec2/(c*t)**3)

    def thermal(
        self,
        t:u.Quantity
    )->u.Quantity:
        return u.Quantity(self.j_scr.thermal*self.j0_quantity(t))

    def powerlaw(
        self,
        t:u.Quantity
    )->u.Quantity:
        return u.Quantity(self.j_scr.powerlaw*self.j0_quantity(t))

    def total(
        self,
        t:u.Quantity
    )->u.Quantity:
        return u.Quantity(self.j0_quantity(t)*self.j_scr.total)

@dataclass(frozen=True,slots=True)
class AbsorptionCoefficient:
    alpha_0: NDArray[np.float64]
    alpha_scr: AbsorptionCoefficientFrequencyDependence

    def alpha0_quantity(
        self,
        t:u.Quantity
    ):
        return u.Quantity(self.alpha_0/(c*t))

    def thermal(
        self,
        t:u.Quantity
    )->u.Quantity:
        return u.Quantity(self.alpha0_quantity(t)*self.alpha_scr.thermal)

    def powerlaw(
        self,
        t:u.Quantity
    )->u.Quantity:
        return u.Quantity(self.alpha0_quantity(t)*self.alpha_scr.powerlaw)

    def total(
        self,
        t:u.Quantity
    )->u.Quantity:
        return u.Quantity(self.alpha0_quantity(t)*self.alpha_scr.total)
