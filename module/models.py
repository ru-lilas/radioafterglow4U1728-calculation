import numpy as np
from module import thermal
from dataclasses import dataclass

@dataclass
class SynchrotronSpectrum:
    theta: float
    chi: np.ndarray

    @property
    def xm(self)->np.ndarray:
        return thermal.convert_xm(
            theta=self.theta,
            chi=self.chi
        )

    @property
    def jnu_th(self)->np.ndarray:
        return thermal.j_th(
            theta=self.theta,
            chi=self.chi
        )

    @property
    def anu_th(self)->np.ndarray:
        return thermal.anu_th_dimless(
            theta=self.theta,
            chi=self.chi,
        )
