from module import plot_utils as pu
from dataclasses import replace

lc_continuous = pu.LineStyleConfig(
    colors=[
        "#FF4B00",
        "#005AFF"
    ],
    linestyle="-",
    linewidth=1.0
)
lc_binned = replace(
    lc_continuous,
    linestyle="-",
    linewidth=2.0
)

