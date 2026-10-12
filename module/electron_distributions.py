"""Electron distributions and correction factors for radiation calculations"""

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray
from scipy import special


def _integrated_number_fraction(
    gamma: NDArray[np.float64],
    gamma_pdf: NDArray[np.float64],
    gamma_lower: float,
) -> NDArray[np.float64]:
    """Integrate each row using existing grid points at or above the lower bound."""
    if not np.isfinite(gamma_lower) or gamma_lower < 1.0:
        raise ValueError("積分下限は1以上の有限値が必要です")
    selected = gamma >= gamma_lower
    if np.count_nonzero(selected) < 2:
        raise ValueError("積分範囲には2点以上のgamma格子点が必要です")
    return np.atleast_1d(
        np.trapezoid(gamma_pdf[:, selected], x=gamma[selected], axis=1)
    )


@dataclass(frozen=True, slots=True)
class MaxwellJuttnerFactor:
    """Maxwell-Juttner correction factor and its high-temperature approximation."""

    exact: NDArray[np.float64]
    hot: NDArray[np.float64]

    @classmethod
    def compute(cls, theta: NDArray[np.float64]) -> Self:
        return cls(
            exact=2.0 * theta**2 / special.kv(2, 1.0 / theta),
            hot=np.ones_like(theta),
        )


@dataclass(frozen=True, slots=True)
class MaxwellJuttnerDistribution:
    """Thermal distribution normalized to the total electron number.

    Shapes: theta (N,), gamma (M,), and gamma_pdf (N, M).
    Each row integrates to 1-f_e over gamma from 1 to infinity.
    """

    theta: NDArray[np.float64]
    gamma: NDArray[np.float64]
    gamma_pdf: NDArray[np.float64]

    @property
    def mean_gamma(self) -> NDArray[np.float64]:
        """Mean Lorentz factor within the thermal component; shape (N,)."""
        z = 1.0 / self.theta
        return special.kve(3, z) / special.kve(2, z) - self.theta

    @property
    def mean_gamma_hot(self) -> NDArray[np.float64]:
        return 3.0 * self.theta

    @property
    def mean_kinetic_gamma(self) -> NDArray[np.float64]:
        return self.mean_gamma - 1.0

    def integrated_number_fraction(
        self, gamma_lower: float = 1.001,
    ) -> NDArray[np.float64]:
        """Number fraction integrated on grid points at or above gamma_lower; shape (N,)."""
        return _integrated_number_fraction(self.gamma, self.gamma_pdf, gamma_lower)

    def integrated_kinetic_energy(
        self, gamma_lower: float = 1.001,
    ) -> NDArray[np.float64]:
        """Kinetic energy/(n_e m_e c^2) integrated over the finite grid; shape (N,).

        Return the contribution per total electron, not per electron in this component.
        """
        weighted_pdf = self.gamma_pdf * (self.gamma[None, :] - 1.0)
        return _integrated_number_fraction(self.gamma, weighted_pdf, gamma_lower)

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
        return cls(theta=theta, gamma=gamma, gamma_pdf=gamma_pdf)


@dataclass(frozen=True, slots=True)
class PowerlawNormalization:
    """Power-law normalization factor for radiation and its hot approximation."""

    exact: NDArray[np.float64]
    hot: NDArray[np.float64]

    @classmethod
    def compute(
        cls,
        p: float,
        theta: NDArray[np.float64],
        gamma_min: NDArray[np.float64],
    ) -> Self:
        pm1 = p - 1.0
        exact = (
            pm1 * gamma_min / (pm1 * gamma_min - pm1 + 1.0)
        ) * (gamma_min / (3.0 * theta))**pm1
        return cls(exact=exact, hot=np.ones_like(theta))


@dataclass(frozen=True, slots=True)
class PowerlawDistribution:
    """Power-law distribution normalized to the total electron number.

    Shapes: gamma_min (N,), gamma (M,), and gamma_pdf (N, M).
    Values below gamma_min are zero; each row integrates to f_e to infinity.
    """

    p_index: float
    f_e: float
    gamma_min: NDArray[np.float64]
    gamma: NDArray[np.float64]
    gamma_pdf: NDArray[np.float64]

    def integrated_number_fraction(
        self, gamma_lower: float = 1.001,
    ) -> NDArray[np.float64]:
        """Number fraction integrated on grid points at or above gamma_lower; shape (N,)."""
        return _integrated_number_fraction(self.gamma, self.gamma_pdf, gamma_lower)

    def integrated_kinetic_energy(
        self, gamma_lower: float = 1.001,
    ) -> NDArray[np.float64]:
        """Kinetic energy/(n_e m_e c^2) integrated over the finite grid; shape (N,).

        Return the contribution per total electron, not per electron in this component.
        """
        weighted_pdf = self.gamma_pdf * (self.gamma[None, :] - 1.0)
        return _integrated_number_fraction(self.gamma, weighted_pdf, gamma_lower)

    @classmethod
    def compute(
        cls,
        p_index: float,
        f_e: float,
        gamma_min: NDArray[np.float64],
        gamma: NDArray[np.float64],
    ) -> Self:
        if gamma_min.ndim != 1 or gamma.ndim != 1:
            raise ValueError("gamma_minとgammaは1次元配列が必要です")
        if not p_index > 1:
            raise ValueError("p_indexは1より大きい値が必要です")
        if not 0 <= f_e <= 1:
            raise ValueError("f_eは0以上1以下が必要です")
        if np.any(gamma_min < 1) or np.any(gamma < 1):
            raise ValueError("gamma_minとgammaは1以上が必要です")

        minimum = gamma_min[:, None]
        gamma_row = gamma[None, :]
        gamma_pdf = np.where(
            gamma_row >= minimum,
            f_e * (p_index - 1.0) / minimum
            * (gamma_row / minimum)**(-p_index),
            0.0,
        )
        return cls(
            p_index=p_index,
            f_e=f_e,
            gamma_min=gamma_min,
            gamma=gamma,
            gamma_pdf=gamma_pdf,
        )


@dataclass(frozen=True, slots=True)
class ElectronDistribution:
    """Thermal and power-law distributions on a shared gamma grid, and their sum."""

    thermal: MaxwellJuttnerDistribution
    powerlaw: PowerlawDistribution

    def __post_init__(self) -> None:
        if not np.array_equal(self.thermal.gamma, self.powerlaw.gamma):
            raise ValueError("thermalとpowerlawのgammaが一致しません")
        if self.thermal.gamma_pdf.shape != self.powerlaw.gamma_pdf.shape:
            raise ValueError("thermalとpowerlawの分布形状が一致しません")

    @property
    def gamma(self) -> NDArray[np.float64]:
        return self.thermal.gamma

    @property
    def gamma_pdf(self) -> NDArray[np.float64]:
        return self.thermal.gamma_pdf + self.powerlaw.gamma_pdf

    def integrated_number_fraction(
        self, gamma_lower: float = 1.001,
    ) -> NDArray[np.float64]:
        """Sum of thermal and power-law number fractions over the same integration range."""
        return (
            self.thermal.integrated_number_fraction(gamma_lower)
            + self.powerlaw.integrated_number_fraction(gamma_lower)
        )

    def integrated_kinetic_energy(
        self, gamma_lower: float = 1.001,
    ) -> NDArray[np.float64]:
        """Sum of thermal and power-law kinetic energies/(n_e m_e c^2)"""
        return (
            self.thermal.integrated_kinetic_energy(gamma_lower)
            + self.powerlaw.integrated_kinetic_energy(gamma_lower)
        )

    @classmethod
    def compute(
        cls,
        theta: NDArray[np.float64],
        gamma: NDArray[np.float64],
        p_index: float,
        f_e: float,
        gamma_min: NDArray[np.float64] | None = None,
    ) -> Self:
        if not 0 <= f_e <= 1:
            raise ValueError("f_eは0以上1以下が必要です")

        thermal = MaxwellJuttnerDistribution.compute(
            f_e=f_e,
            theta=theta,
            gamma=gamma,
        )
        minimum = thermal.mean_gamma if gamma_min is None else gamma_min
        if minimum.shape != theta.shape:
            raise ValueError("gamma_minとthetaは同じ形状が必要です")

        powerlaw = PowerlawDistribution.compute(
            p_index=p_index,
            f_e=f_e,
            gamma_min=minimum,
            gamma=gamma,
        )
        return cls(thermal=thermal, powerlaw=powerlaw)
