from module.utilities import filereaders as fr
from pathlib import Path
import argparse
from module.plot import contour
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from module.plot import plot_utils
from module.utilities.quantity_data import QuantityData
from module.strenums import KeyNames

def annotation_quantity(conf:dict,metadata:dict):
    value_keyname = conf["value"]
    unit_keyname = conf["unit"]
    value = metadata[value_keyname]
    unit = metadata[unit_keyname]
    q_data = QuantityData(
        value=value,
        unit=unit
    )
    prefix = conf["prefix"]
    fmt = conf["fmt"]
    return f"{prefix}{q_data.quantity:{fmt}}"

def main(args:argparse.Namespace):
    inpath:Path = args.input
    scatpath:Path = args.scatters
    confpath:Path = args.config
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    conf = fr.read_yaml_pyyaml(confpath)
    df_contour = fr.read_csv(inpath)
    metadata_contour = fr.read_keyvalue(inpath)
    df_scatters = fr.read_csv(scatpath)

    with PdfPages(outpath) as pdf:
        figsize=conf["figsize"]
        fig,ax = plt.subplots(figsize=figsize)
        fig.set_layout_engine("constrained")

        contour.plot_parameter_curves(
            ax=ax,
            conf=conf,
            df=df_contour,
        )
        contour.plot_scatters(ax,conf["observation_scatters"],df_scatters)

        d_text = annotation_quantity(conf["annotations"]["quantities"]["d"],metadata_contour)
        
        annot = plot_utils.AnnotationConfigure(
            use=True,
            fontsize=16,
            text=\
                r"$\varepsilon_{B}=$"f"{metadata_contour[KeyNames.EPS_B]}"
                r", $\varepsilon_\mathrm{th}=$"f"{metadata_contour[KeyNames.EPS_TH]}"
                r", $\mu=$"f"{metadata_contour['mu']}"
                r", $\mu_e=$"f"{metadata_contour['mu_e']}"
                f", {d_text}"
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
        "-s",
        "--scatters",
        type=Path,
        required=True
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

