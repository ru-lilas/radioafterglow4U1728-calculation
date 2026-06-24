from typing import Any
import numpy as np
from numpy.typing import NDArray

def log(
    arr_conf:dict[str,Any],
)->NDArray[np.float64]:
    return np.logspace(
        **arr_conf,
        dtype=np.float64
    )

def linear(
    arr_conf:dict[str,Any],
)->NDArray[np.float64]:
    return np.linspace(
        **arr_conf,
        dtype=np.float64
    )
