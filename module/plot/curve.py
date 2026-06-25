from matplotlib.axes import Axes
from module.plot.plot_utils import CurveConfigure

def curve(
    ax:Axes,
    x,y,curveconf:CurveConfigure
):
    ax.plot(
        x,y,
        ls=curveconf.linestyle,
        lw=curveconf.linewidth,
        color=curveconf.color,
        label=curveconf.label
    )
