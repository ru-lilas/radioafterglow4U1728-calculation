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
    theta: float
    gamma: NDArray[np.float64]
    gamma_pdf: NDArray[np.float64]

    @property
    def mean_gamma(self)->float:
        z = 1.0/self.theta
        return float(special.kve(3,z)/special.kve(2,z) - self.theta)

    @property
    def mean_gamma_hot(self)->float:
        return 3.0*self.theta

    @property
    def mean_kinetic_gamma(self) -> float:
        return self.mean_gamma - 1.0

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
