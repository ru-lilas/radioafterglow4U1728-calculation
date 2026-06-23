from typing import Any
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
from module.plot import processor
import pandas as pd
from module.plot import plot_utils

def common(
    ax:Axes,
    conf:dict[str,Any],
    df:pd.DataFrame,
):
    commonconf = conf["common"]
    conftick = plot_utils.TicksConfigure(**commonconf["TicksConfigure"])
    labelconf = plot_utils.LabelConfigure(**commonconf["LabelConfigure"])
    plot_utils.configure_tick(ax,conftick)
    plot_utils.configure_label(ax,labelconf)
    x = processor.build_axis_array(commonconf["x"],df)
    y = processor.build_axis_array(commonconf["y"],df)
    return x,y

def colormap(
    fig:Figure,
    ax:Axes,
    x:NDArray[np.float64],
    y:NDArray[np.float64],
    conf:dict[str,Any],
    df:pd.DataFrame,
):
    cmconf = conf["colormap"]
    zconf = cmconf["z"]
    zname = str(zconf["name"])

    z = np.asarray(df[zname],dtype=np.float64)
    im = ax.scatter(
        x,
        y,
        c=z,
        norm=LogNorm() if (zconf["scale"] == "log") else None,
        s=2,
        cmap="cividis_r",
    )
    cbarconf = cmconf["cbar"]
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(
        cbarconf["label"],
        fontsize=cbarconf["fontsize"]
    )
    cbar.ax.tick_params(labelsize=cbarconf["tickfontsize"])
    cbar.locator = LogLocator(base=10)
    cbar.update_ticks()
    return

def contour(
    ax:Axes,
    x:NDArray[np.float64],
    y:NDArray[np.float64],
    conf:dict[str,Any],
    df:pd.DataFrame,
):
    if not "contours" in conf.keys():
        return

    for cname, cconf in conf["contours"].items():
        z = processor.build_axis_array(cconf,df)
        ax.tricontour(x.ravel(),y.ravel(),z.ravel())

    return

def level_curve(
    ax: Axes,
    parameter_conf:dict[str,Any],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    level: float
):
    if parameter_conf["logscale"]:
        mask = np.abs(
            np.log10(z / level)
        ) < parameter_conf["dex_tol"]
    else:
        mask = np.isclose(
            z,
            level,
            rtol=parameter_conf["rtol"],
        )
    idx = np.argsort(x[mask])
    x_curve = np.asarray(x[mask],dtype=np.float64)
    y_curve = np.asarray(y[mask],dtype=np.float64)
    ax.plot(
        x_curve[idx],
        y_curve[idx],
        linestyle=parameter_conf["linestyle"],
        linewidth=parameter_conf["linewidth"],
        color=parameter_conf["color"],
    )
    idx = int(0.5*len(x_curve))

    labelconf = parameter_conf["label"]
    fmt = labelconf["fmt"]
    prefix = labelconf["prefix"]
    suffix = labelconf["suffix"]
    ax.text(
        x_curve[idx],
        y_curve[idx],
        f"{prefix}{level:{fmt}}{suffix}",
        fontsize = labelconf["fontsize"],
        rotation = labelconf["rotation"],
        ha=labelconf["ha"],
        va=labelconf["va"],
        clip_on=True,
        bbox={
            "facecolor": labelconf["facecolor"],
            "edgecolor": "none",
            "pad": labelconf["pad"]
        },
    )
    return

def parameter_curves(
    ax: Axes,
    parameter_conf:dict[str,Any],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
):
    for level in parameter_conf["levels"]:
        level_curve(ax,parameter_conf,x,y,z,level)
    return

def plot_colormaps(
    conf:dict[str,Any],
    df:pd.DataFrame,
    outpath: Path
):

    with PdfPages(outpath) as pdf:

        figsize=(12,12)
        fig,ax = plt.subplots(figsize=figsize)
        fig.set_layout_engine("constrained")
        x,y = common(ax,conf,df)

        colormap(fig,ax,x,y,conf,df)
        pdf.savefig(fig)

        plt.close(fig)

def plot_parameter_curves(
    ax:Axes,
    conf:dict[str,Any],
    df:pd.DataFrame,
):
    x,y = common(ax,conf,df)

    pcconf:dict[str,Any] = conf["parameter_curves"]
    for _, param_conf in pcconf.items():
        column_name = param_conf["column_name"]
        z = np.asarray(df[column_name],dtype=np.float64)
        parameter_curves(
            ax,
            param_conf,
            x,y,z
        )

def plot_scatters(
    ax:Axes,
    conf:dict[str,Any],
    df_scatters:pd.DataFrame,
):
    ax.scatter(df_scatters["phi_peak"],df_scatters["l_peak"])
    return

def plot_contours(
    conf:dict[str,Any],
    df:pd.DataFrame,
    outpath: Path
):
    with PdfPages(outpath) as pdf:

        figsize=(12,12)
        fig,ax = plt.subplots(figsize=figsize)
        fig.set_layout_engine("constrained")
        x,y = common(ax,conf,df)
        contour(ax,x,y,conf,df)
        pdf.savefig(fig)
        plt.close(fig)
