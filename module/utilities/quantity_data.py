from dataclasses import dataclass
from functools import cached_property
from numpy.typing import NDArray
import numpy as np
import astropy.units as u

@dataclass
class QuantityData:
    value: NDArray[np.float64]
    unit: str

    @cached_property
    def quantity(self):
        return u.Quantity(self.value,u.Unit(self.unit))

    def unit_convert(self,unit:str):
        return u.Quantity(self.quantity.to(u.Unit(unit)))

    def to_ndarray(self,unit:str):
        return np.asarray(
            self.quantity.to_value(u.Unit(unit)),
            dtype=np.float64
        )
