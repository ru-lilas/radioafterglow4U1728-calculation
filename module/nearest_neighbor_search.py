import numpy as np
from numpy.typing import NDArray

def linear(
    log10x_bg:NDArray[np.float64],
    log10y_bg:NDArray[np.float64],
    log10x_pt:NDArray[np.float64],
    log10y_pt:NDArray[np.float64]
)->NDArray[np.intp]:
    dx = log10x_bg[:, None] - log10x_pt[None, :]
    dy = log10y_bg[:, None] - log10y_pt[None, :]
    dist2 = dx*dx + dy*dy
    return np.argmin(dist2, axis=0)
