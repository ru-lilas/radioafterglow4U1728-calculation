from module.utilities import filereaders as fr
from pathlib import Path
import argparse
from module.plot import curve
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from module.plot import plot_utils
import numpy as np
from module.plot import plot_scatter
import pandas as pd

from module.utilities import quantity_data

def main(args:argparse.Namespace):
    inpath:Path = args.input
    confpath:Path = args.config
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)
    obspath:Path = args.observation

    metadata_obs = fr.read_keyvalue(obspath)
    df_obs = fr.read_csv(obspath)

    conf = fr.read_yaml(confpath)
    df_calc = fr.read_csv(inpath)
    metadata_calc = fr.read_keyvalue(inpath)

    nu_values:list[float] = conf["nu_values"]
    df_nu = pd.DataFrame(df_calc[np.isclose(df_calc["nu_value"], nu_values[1])].reset_index(drop=True))

    fnu_per = quantity_data.QuantityData(
        value=metadata_obs["f9_per"],
        unit=metadata_obs["fnu_unit"]
    )
    df_nu["fnu_value_add"] = df_nu["fnu_value"] + fnu_per
    
    t_min = curve.build_axisarray(
        df_nu,
        "t_value",
        metadata_calc["t_unit"],
        conf["t_unit"]
    )
    fnu_mjy = curve.build_axisarray(
        df_nu,
        "fnu_value_add",
        metadata_calc["fnu_unit"],
        conf["fnu_unit"]
    )
    curveconf = plot_utils.CurveConfigure(**conf["calculation"])
    curveconf.label = f"{nu_values[1]} {conf['nu_unit']}"

    scatterconf = plot_scatter.ScatterConfigure(**conf["observation_scatter"])
    text = ""
    for key,element in conf["annotation"]["elements"].items():
        value = metadata_calc[key]
        prefix = element["prefix"]
        fmt = element["fmt"]
        text += f"{prefix}={value:{fmt}} "

    annot = plot_utils.AnnotationConfigure(
        **conf["annotation"]["config"],
        text=text
    )

    with PdfPages(outpath) as pdf:
        figsize=conf["figsize"]
        fig,ax = plt.subplots(figsize=figsize)
        fig.set_layout_engine("constrained")

        plot_utils.configure_label(
            ax,
            plot_utils.LabelConfigure(**conf["LabelConfigure"])
        )
        plot_utils.configure_tick(
            ax,
            plot_utils.TicksConfigure(**conf["TicksConfigure"])
        )
        curve.curve(ax,t_min,fnu_mjy,curveconf)
        plot_scatter.with_errorbar(
            ax,
            np.asarray(df_obs["t"]),
            np.asarray(df_obs["f9"]),
            np.asarray(df_obs["t_err"]),
            np.asarray(df_obs["f9_err"]),
            scatterconf
        )
        legend_handles = []
        plot_utils.hlines(ax,conf,metadata_obs,legend_handles)
        plot_utils.annotation(ax,annot)
        ax.legend()

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
        "--observation",
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


