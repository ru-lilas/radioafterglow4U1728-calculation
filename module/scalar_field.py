"""Numerical grid data shared by contour and colormap components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, eq=False, init=False)
class ScalarField2D:
    """A scalar field on a grid, independent of units and drawing styles.

    ``z`` has shape ``(ny, nx)`` with at least two points per axis.
    Coordinates must be either two 1-D arrays of lengths ``nx`` and ``ny``,
    or two 2-D arrays with the same shape as ``z``. One-dimensional
    coordinates must be strictly increasing or strictly decreasing.

    Inputs are copied; coordinates and field data are read-only float64 arrays. Existing
    masks on ``z`` are retained; NaN and infinite field values are masked.
    Coordinates must be finite and unmasked. Convert physical units before
    constructing the field. The dataclass prevents attribute reassignment,
    but the NumPy mask remains writable for Matplotlib compatibility.
    Use ``masked()`` to create independently masked fields.

    The constructor accepts ordinary float64 arrays or masked arrays for
    ``z``; the stored ``z`` is always a MaskedArray.

    Pass ``field.x``, ``field.y``, and ``field.z`` directly to Matplotlib's
    ``Axes.contour`` or ``Axes.pcolormesh``.
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z: np.ma.MaskedArray

    def __init__(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        z: NDArray[np.float64] | np.ma.MaskedArray,
    ) -> None:
        """Copy, normalize, and validate inputs before assigning attributes."""
        x = self._coordinate(x, "x")
        y = self._coordinate(y, "y")
        z = np.ma.array(z, dtype=np.float64, copy=True)

        if z.ndim != 2 or min(z.shape) < 2:
            raise ValueError("z must be a 2-D array with shape (ny, nx), ny >= 2 and nx >= 2")
        if x.ndim == y.ndim == 1:
            if z.shape != (y.size, x.size):
                raise ValueError("For 1-D coordinates, z.shape must equal (y.size, x.size)")
            for name, coordinate in (("x", x), ("y", y)):
                if not (np.all(coordinate[1:] > coordinate[:-1]) or
                        np.all(coordinate[1:] < coordinate[:-1])):
                    raise ValueError(f"{name} must be strictly monotonic")
        elif x.ndim == y.ndim == 2:
            if x.shape != z.shape or y.shape != z.shape:
                raise ValueError("For 2-D coordinates, x, y, and z must have the same shape")
        else:
            raise ValueError("x and y must both be 1-D or both be 2-D")

        z = np.ma.masked_invalid(z, copy=False)
        # Use an explicit mask, including when all locations are valid.
        z.mask = np.ma.getmaskarray(z).copy()
        z.setflags(write=False)
        x.setflags(write=False)
        y.setflags(write=False)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "z", z)

    @staticmethod
    def _coordinate(values: NDArray[np.float64], name: str) -> NDArray[np.float64]:
        if np.any(np.ma.getmaskarray(values)):
            raise ValueError(f"{name} must not contain masked coordinates")
        coordinate = np.array(values, dtype=np.float64, copy=True)
        if not np.all(np.isfinite(coordinate)):
            raise ValueError(f"{name} must contain only finite coordinates")
        return coordinate

    def masked(self, mask: NDArray[np.bool_]) -> ScalarField2D:
        """Return a new field with additional locations excluded.

        ``True`` means excluded. The Boolean mask must have exactly the
        same shape as ``z``; broadcasting is intentionally not supported.
        Existing exclusions are combined using logical OR. Neither this
        field nor the supplied mask is modified.
        """
        mask_array = np.asarray(mask)
        if mask_array.dtype != np.dtype(np.bool_):
            raise TypeError("mask must contain Boolean values")
        if mask_array.shape != self.z.shape:
            raise ValueError("mask.shape must equal z.shape")
        combined_mask = np.ma.getmaskarray(self.z) | mask_array
        return ScalarField2D(
            x=self.x,
            y=self.y,
            z=np.ma.array(self.z, mask=combined_mask, copy=True),
        )
