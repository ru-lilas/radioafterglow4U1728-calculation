import matplotlib.pyplot as plt
from module.utilities import filereaders as fr
from module.utilities import plot_utils
from matplotlib.lines import Line2D
from pathlib import Path
import argparse
from matplotlib.backends.backend_pdf import PdfPages
import astropy.units as u

def main(args:argparse.Namespace):
    inpath:Path = args.input
    confpath:Path = args.config
    outpath:Path = args.output

    conf = fr.read_yaml(confpath)
    conftick = plot_utils.TicksConfigure(**conf["TicksConfigure"])
    conflabel = plot_utils.LabelConfigure(**conf["LabelConfigure"])
    xname = conf["x_name"]
    yname = conf["y_name"]


    df = fr.read_csv(inpath)
    metadata = fr.read_keyvalue(inpath)
    outpath.parent.mkdir(parents=True,exist_ok=True)

    t_value_ref = metadata["t_peak_ref"]
    t_unit_ref = "s"
    t_ref = u.Quantity(t_value_ref,u.Unit(t_unit_ref))

    nu_ref = metadata["nu_peak_ref"]
    lnu_peak_estimated = metadata["lnu_peak_ref"]

    with PdfPages(outpath) as pdf:
    
        figsize=(16,9)
        fig,ax = plt.subplots(figsize=figsize)
        fig.set_layout_engine("constrained")
        plot_utils.configure_tick(ax,conftick)
        plot_utils.configure_label(ax,conflabel)
        ax.loglog(
            df[xname],
            df[yname],
            color = "#000000",
            ls="-",
        )

        # reference frequency
        ax.axvline(x=nu_ref,ls="-.",color="#000000")

        # estimated peak-luminosity
        ax.axhline(y=lnu_peak_estimated,ls="--",color="#000000")

        legend_handles = [
            Line2D(
                [0],[0],
                color="#000000",
                ls="-.",
                label=r"$\nu_{\mathrm{ref}}="f"{nu_ref:.1e}$ Hz"
            ),
            Line2D(
                [0],[0],
                color="#000000",
                ls="--",
                label=r"$L_{\nu,\mathrm{peak}}=$"f"{lnu_peak_estimated:.2e} erg/s/Hz"
            )
        ]
        ax.legend(handles=legend_handles,fontsize=16)

        annotconf = plot_utils.AnnotationConfigure(
            use=True,
            fontsize=16,
            text = \
            r"$t=$"f"{t_ref:.1e}"
        )
        plot_utils.annotation(ax,annotconf)

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

