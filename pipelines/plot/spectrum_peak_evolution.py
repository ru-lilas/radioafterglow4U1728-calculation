import matplotlib.pyplot as plt
from module.utilities import filereaders as fr
from module.utilities import plot_utils
from pathlib import Path
import argparse
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

def main(args:argparse.Namespace):
    inpath:Path = args.input
    confpath:Path = args.config
    outpath:Path = args.output

    conf = fr.read_yaml(confpath)
    conftick = plot_utils.TicksConfigure(**conf["TicksConfigure"])
    conflabel = plot_utils.LabelConfigure(**conf["LabelConfigure"])

    outpath.parent.mkdir(parents=True,exist_ok=True)
    with PdfPages(outpath) as pdf:
        df = fr.read_csv(inpath)
    
        figsize=(8,6)
        fig,ax = plt.subplots(figsize=figsize)
        fig.set_layout_engine("constrained")
        plot_utils.configure_tick(ax,conftick)
        plot_utils.configure_label(ax,conflabel)
        ax.loglog(
            df["t_value"],
            df["lnu_peak"],
            color = "#000000",
            ls="-",
        )

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

