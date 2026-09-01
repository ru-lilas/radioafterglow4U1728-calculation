import numpy as np
from dataclasses import dataclass

from module.mydataclasses import QuantityArray
from module.types import FloatArray

@dataclass(frozen=True,slots=True)
class ABGrid:
    a_wind: QuantityArray
    beta_sh: FloatArray

    def build_meshgrid(self):
        return np.meshgrid(
            self.a_wind.values,
            self.beta_sh,
            indexing="xy"
        )
