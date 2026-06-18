import matplotlib.pyplot as plt
from module.utilities import filereaders as fr
from module.utilities import plot_utils
from pathlib import Path
import argparse
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator

def main(args:argparse.Namespace):
    inpath:Path = args.input
    confpath:Path = args.config
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    conf = fr.read_yaml(confpath)
    df = fr.read_csv(inpath)

    conftick = plot_utils.TicksConfigure(**conf["TicksConfigure"])
    conflabel = plot_utils.LabelConfigure(**conf["LabelConfigure"])
    xname = str(conf["x_name"])
    yname = str(conf["y_name"])
    zname = str(conf["z_name"])

    x = np.asarray(df[xname],dtype=np.float64)
    y = np.asarray(df[yname],dtype=np.float64)
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
            norm=LogNorm(),
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


