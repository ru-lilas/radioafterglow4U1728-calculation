from numpy.typing import NDArray
from typing import Literal
import numpy as np

type ValueScale = Literal["linear","log"]
type FloatArray = NDArray[np.float64]
type FloatArrayLike = float|FloatArray
type FloatPair = tuple[float,float]
type FloatGrid = np.ndarray[
    tuple[int, int],
    np.dtype[np.float64],
]
