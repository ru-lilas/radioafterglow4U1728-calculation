# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from module.utilities import filereaders as fr
from module.utilities import plot_utils
from pathlib import Path
import argparse
import numpy as np
from module import build_lightcurve_data
from matplotlib.backends.backend_pdf import PdfPages

def main(args:argparse.Namespace):
    inpath_list:list[Path] = args.input
    confpath:Path = args.config
    outpath:Path = args.output

    conf = fr.read_yaml(confpath)
    conftick = plot_utils.TicksConfigure(**conf["TicksConfigure"])
    conflabel = plot_utils.LabelConfigure(**conf["LabelConfigure"])
    nu_refs = build_lightcurve_data.create_reference_frequency_list(conf["frequency"])

    outpath.parent.mkdir(parents=True,exist_ok=True)
    with PdfPages(outpath) as pdf:
        for nu_ref in nu_refs:
            df = build_lightcurve_data.build_lightcurve_data(inpath_list,nu_ref)
            nu = np.asarray(df["nu"]).mean()
            print(df)
    
            figsize=(8,6)
            fig,ax = plt.subplots(figsize=figsize)
            fig.set_layout_engine("constrained")
            plot_utils.configure_tick(ax,conftick)
            plot_utils.configure_label(ax,conflabel)
            ax.loglog(
                df["t_value"],
                df["lnu"],
                color = "#000000",
                ls="-",
            )
            legend_handles = [
                Line2D([0],[0], color="#000000", ls="-", label=rf"$\nu={nu:.1e}$ Hz"),
            ]
            ax.legend(handles=legend_handles)

            pdf.savefig(fig)
            plt.close(fig)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "input",
        type=Path,
        nargs="*"
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
