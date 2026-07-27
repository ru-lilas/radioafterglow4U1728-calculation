from module.utilities import filereaders as fr
from pathlib import Path
import argparse
from module.plot import contour
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from module.plot import plot_utils

def main(args:argparse.Namespace):
    inpath:Path = args.input
    scatpath:Path = args.scatters
    confpath:Path = args.config
    estpath:Path = args.estimated
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    conf = fr.read_yaml_pyyaml(confpath)
    df_contour = fr.read_csv(inpath)
    metadata_contour = fr.read_keyvalue(inpath)
    df_scatters = fr.read_csv(scatpath)
    df_estimated = fr.read_csv(estpath)

    with PdfPages(outpath) as pdf:
        figsize=conf["figsize"]
        fig,ax = plt.subplots(figsize=figsize)
        fig.set_layout_engine("constrained")

        contour.plot_parameter_curves(
            ax=ax,
            conf=conf,
            df=df_contour,
        )
        contour.plot_scatters(ax,conf,df_scatters)
        ax.scatter(
            df_estimated["phi_peak_est"],
            df_estimated["l_peak_est"],
            color="red",
            s=2
        )

        annot = plot_utils.AnnotationConfigure(
            use=True,
            fontsize=16,
            text=\
                r"$\varepsilon_{B}=$"f"{metadata_contour['eps_B']}"
                r", $\varepsilon_\mathrm{th}=$"f"{metadata_contour['eps_th']}"
                r", $\mu=$"f"{metadata_contour['mu']}"
                r", $\mu_e=$"f"{metadata_contour['mu_e']}"
        )
        plot_utils.annotation(ax,annot)


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
        "--scatters",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--estimated",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True
    )
    
    args = parser.parse_args()
    
    main(args)


