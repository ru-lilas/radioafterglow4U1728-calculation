import matplotlib.pyplot as plt
from numpy.typing import NDArray
import pandas as pd
from module.utilities import filereaders as fr
from module.utilities import plot_utils
from pathlib import Path
import argparse
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator

def revive_quantity_array(
    dimensionless_value: NDArray[np.float64],
    quantity_as_unit: NDArray[np.float64]
)->NDArray[np.float64]:
    return dimensionless_value*quantity_as_unit

def fetch_reviving_quantity(
    df: pd.DataFrame,
    name_x_dimless: str,
    name_x_norm: str
)->tuple[NDArray[np.float64],NDArray[np.float64]]:
    dimless = np.asarray(df[name_x_dimless],dtype=np.float64)
    norm = np.asarray(df[name_x_norm],dtype=np.float64)
    return dimless,norm

def calculate_phi_peak(
    conf: dict,
    df: pd.DataFrame
)->NDArray[np.float64]:
    name_xm_peak = str(conf["name_x_dimensionless"])
    name_phi_theta = str(conf["name_x_normalized"])
    xm_peak,phi_theta = fetch_reviving_quantity(
        df=df,
        name_x_dimless=name_xm_peak,
        name_x_norm=name_phi_theta
    )
    return revive_quantity_array(xm_peak,phi_theta)

def calculate_lnu_peak(
    conf: dict,
    df: pd.DataFrame
)->NDArray[np.float64]:
    name_lnu_peak = str(conf["name_y_dimensionless"])
    name_l_theta = str(conf["name_y_normalized"])
    lnu_peak,l_theta = fetch_reviving_quantity(
        df=df,
        name_x_dimless=name_lnu_peak,
        name_x_norm=name_l_theta
    )
    return revive_quantity_array(lnu_peak,l_theta)

def main(args:argparse.Namespace):
    inpath:Path = args.input
    confpath:Path = args.config
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    conf = fr.read_yaml(confpath)
    df = fr.read_csv(inpath)

    conftick = plot_utils.TicksConfigure(**conf["TicksConfigure"])
    conflabel = plot_utils.LabelConfigure(**conf["LabelConfigure"])
    zname = str(conf["z_name"])

    x = calculate_phi_peak(conf,df)
    y = calculate_lnu_peak(conf,df)
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

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "input",
        type=Path,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True
    )
    
    args = parser.parse_args()
    
    main(args)


