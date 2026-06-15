from typing import Any
import matplotlib.pyplot as plt
from module.utilities import plot_utils
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
from module.plot import processor
import pandas as pd

def contour(
    conf:dict[str,Any],
    df:pd.DataFrame,
    outpath: Path
):
    conftick = plot_utils.TicksConfigure(**conf["TicksConfigure"])
    conflabel = plot_utils.LabelConfigure(**conf["LabelConfigure"])
    zname = str(conf["z_name"])

    x = processor.build_axis_array(conf,df,"x")
    y = processor.build_axis_array(conf,df,"y")
    z = np.asarray(df[zname],dtype=np.float64)

    with PdfPages(outpath) as pdf:

        figsize=(12,12)
        fig,ax = plt.subplots(figsize=figsize)
        fig.set_layout_engine("constrained")
        plot_utils.configure_tick(ax,conftick)
        plot_utils.configure_label(ax,conflabel)

        im = ax.scatter(
            x,
            y,
            c=z,
            norm=LogNorm() if (conf["zscale"] == "log") else None,
            s=2,
            cmap="cividis_r",
        )
        confcbar = conf["cbar"]
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(
            confcbar["label"],
            fontsize=confcbar["fontsize"]
        )
        cbar.ax.tick_params(labelsize=confcbar["tickfontsize"])
        cbar.locator = LogLocator(base=10)
        cbar.update_ticks()
        pdf.savefig(fig)

        plt.close(fig)
