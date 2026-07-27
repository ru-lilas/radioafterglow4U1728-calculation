import matplotlib.pyplot as plt
from module.utilities import filereaders as fr
from module.utilities import plot_utils
from pathlib import Path
import argparse
from matplotlib.backends.backend_pdf import PdfPages

def create_fig(pdf,df,conf):
    conftick = plot_utils.TicksConfigure(**conf["TicksConfigure"])
    conflabel = plot_utils.LabelConfigure(**conf["LabelConfigure"])
    xname = str(conf["x_name"])
    yname = str(conf["y_name"])

    figsize=(16,9)
    fig,ax = plt.subplots(figsize=figsize)
    fig.set_layout_engine("constrained")
    plot_utils.configure_tick(ax,conftick)
    plot_utils.configure_label(ax,conflabel)
    ax.scatter(
        df[xname],
        df[yname],
        color = "#000000",
    )

    pdf.savefig(fig)
    plt.close(fig)

def main(args:argparse.Namespace):
    inpath:Path = args.input
    confpath:Path = args.config
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    df = fr.read_csv(inpath)
    conf = fr.read_yaml_pyyaml(confpath)



    with PdfPages(outpath) as pdf:
        create_fig(pdf,df,conf["fig_nu_err"])
        create_fig(pdf,df,conf["fig_lnu_err"])

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


