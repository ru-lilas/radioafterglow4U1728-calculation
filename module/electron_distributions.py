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
