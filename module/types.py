from dataclasses import dataclass
import numpy.typing as npt
from typing import Literal
import numpy as np

type ValueScale = Literal["linear","log"]
type FloatArray = npt.NDArray[np.float64]
type FloatArrayLike = float|FloatArray
type FloatPair = tuple[float,float]
type FloatGrid = np.ndarray[
    tuple[int, int],
    np.dtype[np.float64],
]
type BoolGrid = npt.NDArray[np.bool_]

@dataclass(frozen=True, slots=True)
class FloatArray1D:
    values: FloatArray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)

        if values.ndim != 1:
            raise ValueError(
                f"values must be 1-dimensional, got shape={values.shape}"
            )

        object.__setattr__(self, "values", values)
