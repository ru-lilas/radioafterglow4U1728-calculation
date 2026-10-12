from typing import Self
from dataclasses import dataclass
from numpy.typing import NDArray
from scipy import special
import numpy as np

@dataclass(frozen=True,slots=True)
class MaxwellJuttnerFactor:
    exact: NDArray[np.float64]
    hot: NDArray[np.float64]

    @staticmethod
    def __compute_exact(
        theta:NDArray[np.float64]
    )->NDArray[np.float64]:
        return (
            2.0*theta**2 / special.kv(2,1.0/theta)
        )

    @staticmethod
    def __compute_hot(
        theta:NDArray[np.float64]
    )->NDArray[np.float64]:
        theta_val = theta
        return np.ones_like(theta_val)

    @classmethod
    def compute(
        cls,
        theta:NDArray[np.float64],
    )->Self:
        return cls(
            exact = cls.__compute_exact(theta),
            hot   = cls.__compute_hot(theta)
        )

@dataclass(frozen=True,slots=True)
class MaxwellJuttnerDistribution:
    theta: NDArray[np.float64]
    gamma: NDArray[np.float64]
    gamma_pdf: NDArray[np.float64]

    @property
    def mean_gamma(self)->NDArray[np.float64]:
        z = 1.0/self.theta
        return (special.kve(3,z)/special.kve(2,z) - self.theta)

    @property
    def mean_gamma_hot(self)->NDArray[np.float64]:
        return 3.0*self.theta

    @property
    def mean_kinetic_gamma(self) -> NDArray[np.float64]:
        return self.mean_gamma - 1.0

    @classmethod
    def compute(
        cls,
        f_e: float,
        theta: NDArray[np.float64],
        gamma: NDArray[np.float64],
    ) -> Self:
        if theta.ndim != 1 or gamma.ndim != 1:
            raise ValueError("thetaとgammaは1次元配列が必要です")
        if np.any(theta <= 0):
            raise ValueError("thetaは正の値が必要です")
        if np.any(gamma < 1):
            raise ValueError("gammaは1以上が必要です")

        theta_column = theta[:, None]
        gamma_row = gamma[None, :]
        f_factor = MaxwellJuttnerFactor.compute(theta_column)

        gamma_pdf = (
            (1.0 - f_e)
            * f_factor.exact
            * np.sqrt(1.0 - gamma_row**(-2))
            * gamma_row**2
            / (2.0 * theta_column**3)
            * np.exp(-gamma_row / theta_column)
        )
        return cls(
            theta=theta,
            gamma=gamma,
            gamma_pdf=gamma_pdf,
        )

@dataclass(frozen=True,slots=True)
class PowerlawNormalization:
    exact: NDArray[np.float64]
    hot: NDArray[np.float64]

    @staticmethod
    def _compute_hot(
        theta:NDArray[np.float64]
    )->NDArray[np.float64]:
        return np.ones_like(theta)

    @staticmethod
    def _compute_exact(
        p: float,
        theta:NDArray[np.float64],
        gamma_min: NDArray[np.float64]
    )->NDArray[np.float64]:
        pm1 = p-1.0
        return (
            pm1*gamma_min/(pm1*gamma_min-pm1+1.0)
        )*(gamma_min/(3.0*theta))**pm1

    @classmethod
    def compute(
        cls,
        p:float,
        theta:NDArray[np.float64],
        gamma_min: NDArray[np.float64]
    )->Self:
        return cls(
            exact = cls._compute_exact(p,theta,gamma_min),
            hot   = cls._compute_hot(theta)
        )

@dataclass(frozen=True, slots=True)
class PowerlawDistribution:
    p_index: float
    f_e: float
    gamma_min: NDArray[np.float64]
    gamma: NDArray[np.float64]
    gamma_pdf: NDArray[np.float64]

    @classmethod
    def compute(
        cls,
        p_index: float,
        f_e: float,
        gamma_min: NDArray[np.float64],
        gamma: NDArray[np.float64],
    ) -> Self:
        if gamma_min.ndim != 1 or gamma.ndim != 1:
            raise ValueError("gamma_minとgammaは1次元配列が必要です")
        if p_index <= 1:
            raise ValueError("p_indexは1より大きい値が必要です")
        if not 0 <= f_e <= 1:
            raise ValueError("f_eは0以上1以下が必要です")
        if np.any(gamma_min < 1) or np.any(gamma < 1):
            raise ValueError("gamma_minとgammaは1以上が必要です")

        minimum = gamma_min[:, None]
        gamma_row = gamma[None, :]

        gamma_pdf = np.where(
            gamma_row >= minimum,
            f_e * (p_index - 1.0) / minimum
            * (gamma_row / minimum)**(-p_index),
            0.0,
        )
        return cls(
            p_index=p_index,
            f_e=f_e,
            gamma_min=gamma_min,
            gamma=gamma,
            gamma_pdf=gamma_pdf,
        )

@dataclass(frozen=True, slots=True)
class ElectronDistribution:
    thermal: MaxwellJuttnerDistribution
    powerlaw: PowerlawDistribution

    def __post_init__(self) -> None:
        if not np.array_equal(self.thermal.gamma, self.powerlaw.gamma):
            raise ValueError("thermalとpowerlawのgammaが一致しません")
        if self.thermal.gamma_pdf.shape != self.powerlaw.gamma_pdf.shape:
            raise ValueError("thermalとpowerlawの分布形状が一致しません")

    @property
    def gamma(self) -> NDArray[np.float64]:
        return self.thermal.gamma

    @property
    def gamma_pdf(self) -> NDArray[np.float64]:
        return self.thermal.gamma_pdf + self.powerlaw.gamma_pdf

    @classmethod
    def compute(
        cls,
        theta: NDArray[np.float64],
        gamma: NDArray[np.float64],
        p_index: float,
        f_e: float,
        gamma_min: NDArray[np.float64] | None = None,
    ) -> Self:
        if not 0 <= f_e <= 1:
            raise ValueError("f_eは0以上1以下が必要です")

        thermal = MaxwellJuttnerDistribution.compute(
            f_e=f_e,
            theta=theta,
            gamma=gamma,
        )

        minimum = thermal.mean_gamma if gamma_min is None else gamma_min
        if minimum.shape != theta.shape:
            raise ValueError("gamma_minとthetaは同じ形状が必要です")

        powerlaw = PowerlawDistribution.compute(
            p_index=p_index,
            f_e=f_e,
            gamma_min=minimum,
            gamma=gamma,
        )
        return cls(
            thermal=thermal,
            powerlaw=powerlaw,
        )
