"""Dimensionless thermal synchrotron luminosity and peak extraction."""

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DimensionlessThermalLuminosity:
    """Thermal luminosity with shape (n_xm, n_tau), without model normalization."""

    tau_th: NDArray[np.float64]
    xm: NDArray[np.float64]
    lambda_th: NDArray[np.float64]

    @staticmethod
    def __compute_lambda_th(
        tau_th: NDArray[np.float64],
        xm:NDArray[np.float64],
        ip:NDArray[np.float64]
    )->NDArray[np.float64]:
        if tau_th.ndim != 1 or xm.ndim != 1:
            raise ValueError("tau_thとxmは1次元配列が必要です")
        if ip.ndim != 1 or ip.shape != xm.shape:
            raise ValueError("ipはxmと同じ形状の1次元配列が必要です")
        if xm.size == 0 or tau_th.size == 0:
            raise ValueError("xmとtau_thは空でない配列が必要です")
        xm_column = xm[:, None]
        tau_row = tau_th[None, :]
        ip_column = ip[:, None]
        return (
            9.0*xm_column**2/(8.0*np.pi**2)
            *(-np.expm1(-tau_row*ip_column/xm_column))
        )

    @classmethod
    def compute(
        cls,
        tau_th: NDArray[np.float64],
        xm:NDArray[np.float64],
        ip:NDArray[np.float64]
    )->Self:
        lambda_th = cls.__compute_lambda_th(
            tau_th,xm,ip
        )
        return cls(
            tau_th=tau_th,
            xm=xm,
            lambda_th=lambda_th
        )

    def peaks(self) -> NDArray[np.float64]:
        """Return (peak xm, peak luminosity) for each tau_th; shape (n_tau, 2)."""
        indices = np.argmax(self.lambda_th, axis=0)
        return np.column_stack((
            self.xm[indices],
            self.lambda_th[indices, np.arange(len(indices))]
        ))
