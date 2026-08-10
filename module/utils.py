from dataclasses import dataclass
import numpy as np
from typing import Callable
from module.types import FloatArray

class Integrator:
    @staticmethod
    def trapezoid(
        x: FloatArray,
        y: FloatArray,
    ) -> float:
        return float(np.trapezoid(y, x))
