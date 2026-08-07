from dataclasses import dataclass
import numpy as np
from scipy.special import roots_laguerre,roots_legendre
from typing import Callable
from module.types import FloatArray

class Integrator:
    @staticmethod
    def trapezoid(
        x: FloatArray,
        y: FloatArray,
    ) -> float:
        return float(np.trapezoid(y, x))
