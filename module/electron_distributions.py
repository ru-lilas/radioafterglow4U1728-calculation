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
