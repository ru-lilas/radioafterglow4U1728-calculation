
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from module.utilities import filereaders as fr
from module.utilities import plot_utils
from matplotlib.lines import Line2D
from pathlib import Path
import argparse
from matplotlib.backends.backend_pdf import PdfPages
import astropy.units as u
import pandas as pd

def inneranotation_peak_errors(
    ax:Axes,
    conf:dict,
    metadata: dict
):
    if "peak_errors" not in conf.keys():
        return
    else:
        conf_peak_errors:dict = conf["peak_errors"]
        text = ""
        if "x" in conf_peak_errors:
            conf_peak_error = conf_peak_errors["x"]
            prefix = str(conf_peak_error["prefix"])
            fmt = str(conf_peak_error["fmt"])
            denom_key = conf_peak_error["denom_key"]
            numer_key = conf_peak_error["numer_key"]
            denom = float(metadata[denom_key])
            numer = float(metadata[numer_key])
            err = 1.0 - numer/denom
            text = prefix+f"{err:{fmt}}"
        if "y" in conf_peak_errors:
            conf_peak_error = conf_peak_errors["y"]
            prefix = str(conf_peak_error["prefix"])
            fmt = str(conf_peak_error["fmt"])
            denom_key = conf_peak_error["denom_key"]
            numer_key = conf_peak_error["numer_key"]
            denom = float(metadata[denom_key])
            numer = float(metadata[numer_key])
            err = 1.0 - numer/denom
            text += ("\n"+prefix+f"{err:{fmt}}")
        annotconf = plot_utils.AnnotationConfigure(
            use=True,
            fontsize=16,
            text = text,
            pos=(0.01,0.99),
            ha="left",
            va="top"
        )
        plot_utils.annotation(ax,annotconf)

    return

def plot_debag_value(
    pdf:PdfPages,
    conf:dict,
    df:pd.DataFrame,
    t_ref:u.Quantity,
    metadata:dict
)->None:

    legend_handles:list[Line2D] = []

    conftick = plot_utils.TicksConfigure(**conf["TicksConfigure"])
    conflabel = plot_utils.LabelConfigure(**conf["LabelConfigure"])
    xname = str(conf["x_name"])
    yname = str(conf["y_name"])

    figsize=(16,9)
    fig,ax = plt.subplots(figsize=figsize)
    fig.set_layout_engine("constrained")
    plot_utils.configure_tick(ax,conftick)
    plot_utils.configure_label(ax,conflabel)
    ax.plot(
        df[xname],
        df[yname],
        color = "#000000",
        ls="-",
    )
    plot_utils.hlines(ax,conf,metadata,legend_handles)
    if "vlines" in conf.keys():
        conf_vlines:dict = conf["vlines"]
        for key,conf_vline in conf_vlines.items():
            if key in metadata.keys():
                ax.axvline(
                    metadata[key],
                    color = conf_vline["color"],
                    ls=conf_vline["ls"],
                )
            else:
                ax.axvline(
                    conf_vline["value"],
                    color = conf_vline["color"],
                    ls=conf_vline["ls"],
                )
    ax.legend(handles=legend_handles,fontsize=16)
            
    t_ref = metadata["t"]
    t_unit = metadata["t_unit"]
    t = u.Quantity(t_ref,u.Unit(t_unit))
    phi_theta_value = metadata["phi_theta"]
    phi_unit = metadata["phi_unit"]
    phi_theta = u.Quantity(phi_theta_value,u.Unit(phi_unit))
    l_theta_value = metadata["l_theta"]
    l_unit = metadata["l_unit"]
    l_theta = u.Quantity(l_theta_value,u.Unit(l_unit))

    # if "annotations" in conf.keys():
    #     conf_annot:dict = conf["annotations"]
    #     for quantity_name, quantity_setting in conf_annot.items():
    #         value = metadata[quantity_name]
    #         unit_keyname = quantity_setting["unit"]
    #         unit = metadata[unit_keyname]
    #         quantity = u.Quantity(value,u.Unit(unit))
    #         text_prefix = str(quantity_setting["prefix"])
    #         text_element = text_prefix + f"{quantity}:.1e"

    # if "inner_annotation" in conf.keys():
    #     conf_inneranot:dict = conf["inner_annotation"]
    #
    #     for key,conf_vline in conf_inneranot.items():
    #         if key in metadata.keys():
    #             value = float(metadata[key])
    #             prefix = str(conf_vline["prefix"])
    #             fmt = str(conf_vline["fmt"])
    #             text = prefix+f"{value:{fmt}}"
    #         else:
    #             text = ""
    #         annotconf = plot_utils.AnnotationConfigure(
    #             use=True,
    #             fontsize=16,
    #             text = text,
    #             pos=(0.01,0.99),
    #             ha="left",
    #             va="top"
    #         )
    #         plot_utils.annotation(ax,annotconf)
    inneranotation_peak_errors(ax,conf,metadata)

    annotconf = plot_utils.AnnotationConfigure(
        use=True,
        fontsize=16,
        text = \
        r"$t=$"f"{t:.1e}, " \
        r'$\phi_{\Theta}=$'f"{phi_theta:.1e}, " \
        r'$L_{\Theta}=$'f"{l_theta:.1e}"
    )
    plot_utils.annotation(ax,annotconf)

    pdf.savefig(fig)

    plt.close(fig)
    return

def main(args:argparse.Namespace):
    inpath:Path = args.input
    confpath:Path = args.config
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    conf = fr.read_yaml_pyyaml(confpath)
    df = fr.read_csv(inpath)

    metadata = fr.read_keyvalue(inpath)
    # nu_ref = metadata["nu_peak_ref"]
    # lnu_peak_estimated = metadata["lnu_peak_ref"]
    #
    t_value_ref = metadata["t"]
    t_unit_ref = "s"
    t_ref = u.Quantity(t_value_ref,u.Unit(t_unit_ref))
    #
    # conftick = plot_utils.TicksConfigure(**conf["TicksConfigure"])
    # conflabel = plot_utils.LabelConfigure(**conf["LabelConfigure"])
    # xname = str(conf["x_name"])
    # yname = str(conf["y_name"])
    #
    # # peak luminosity
    # row_peak = pd.DataFrame(df.loc[[df[yname].idxmax()]])
    # nu_peak = float(row_peak.iloc[0][xname])
    # lnu_peak = float(row_peak.iloc[0][yname])
    # nu_err = 1.0 - nu_peak/nu_ref
    # lnu_err = 1.0 - lnu_peak/lnu_peak_estimated

    with PdfPages(outpath) as pdf:

        # figsize=(16,9)
        # fig,ax = plt.subplots(figsize=figsize)
        # fig.set_layout_engine("constrained")
        # plot_utils.configure_tick(ax,conftick)
        # plot_utils.configure_label(ax,conflabel)
        # ax.loglog(
        #     df[xname],
        #     df[yname],
        #     color = "#000000",
        #     ls="-",
        # )
        #
        # # reference frequency
        # ax.axvline(x=nu_ref,ls="-.",color="#000000")
        #
        # # estimated peak-luminosity
        # ax.axhline(y=lnu_peak_estimated,ls="--",color="#000000")
        #
        # legend_handles = [
        #     Line2D(
        #         [0],[0],
        #         color="#000000",
        #         ls="-.",
        #         label=r"$\nu_{\mathrm{ref}}=$"f"{nu_ref:.1e} Hz"
        #     ),
        #     Line2D(
        #         [0],[0],
        #         color="#000000",
        #         ls="--",
        #         label=r"$L_{\nu,\mathrm{est}}=$"f"{lnu_peak_estimated:.2e} erg/s/Hz"
        #     )
        # ]
        # ax.legend(handles=legend_handles,fontsize=16)
        #
        # annotconf = plot_utils.AnnotationConfigure(
        #     use=True,
        #     fontsize=16,
        #     text = \
        #     r"$t=$"f"{t_ref:.1e}" \
        # )
        # plot_utils.annotation(ax,annotconf)
        #
        # plot_utils.annotation(
        #     ax,
        #     plot_utils.AnnotationConfigure(
        #         use=True,
        #         fontsize=16,
        #         pos=(0.01,0.99),
        #         ha="left",va="top",
        #         text= \
        #             r"$1-\nu_{\mathrm{peak}}~/~\nu_{\mathrm{ref}}=$"f"{nu_err:.2e}\n" \
        #             r"$1-L_{\nu,{\mathrm{peak}}}~/~L_{\nu,{\mathrm{est}}}=$"f"{lnu_err:.2e}"
        # ))
        #
        #
        # pdf.savefig(fig)
        #
        # plt.close(fig)

        plot_debag_value(pdf,conf["plot_lnudimless_xi"],df,t_ref,metadata)
        plot_debag_value(pdf,conf["ln_tau"],df,t_ref,metadata)
        plot_debag_value(pdf,conf["f_esc"],df,t_ref,metadata)

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

