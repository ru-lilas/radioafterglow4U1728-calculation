from dataclasses import dataclass
import numpy as np
from scipy.special import roots_laguerre
from typing import Callable

@dataclass
class GaussLaguerreIntegrator:
    """

    Attributes: 
        n_points: 
        x: array of integration variables 
        w: 
    """
    n_points: int
    x: np.ndarray|None = None
    w: np.ndarray|None = None

    def __post_init__(self):
        self.x, self.w = roots_laguerre(self.n_points)

    def integrate(self, func:Callable[[np.ndarray],np.ndarray]):
        if self.x is None or self.w is None:
             raise RuntimeError("Integrator is not initialized properly.")
        else:
            return np.sum(self.w * func(self.x))
