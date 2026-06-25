from matplotlib.axes import Axes
from dataclasses import dataclass

from numpy.typing import NDArray
import numpy as np

@dataclass
class ScatterConfigure:
    size: float
    color: str
    marker: str = "o"

def scatter_only(
    ax:Axes,
    x:NDArray[np.float64],
    y:NDArray[np.float64],
    conf:ScatterConfigure
):
    ax.scatter(
        x,y,
        s=conf.size,
        c=conf.color,
        marker=conf.marker
    )
    return
