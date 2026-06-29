from typing import cast
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
    path_calclc:Path = args.calculation_lc
    confpath:Path = args.config
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)
    path_obslc:Path = args.observation_lc

    metadata_obs = fr.read_keyvalue(path_obslc)
    df_obs_raw = fr.read_csv(path_obslc)

    conf = fr.read_yaml(confpath)
    df_calc_raw = fr.read_csv(path_calclc)
    metadata_calc = fr.read_keyvalue(path_calclc)

    for nu_obs, df_obs in df_obs_raw.groupby("nu",sort=False):
        df_obs = df_obs.reset_index(drop=True)
        nu_obs = cast(float,nu_obs)
        df_nu = pd.DataFrame(df_calc_raw[np.isclose(df_calc_raw["nu"], nu_obs)].reset_index(drop=True))

        fnu_bg = quantity_data.QuantityData(
            value=np.asarray(df_obs["fnu_bg"],dtype=np.float64)[0],
            unit=metadata_obs["fnu_unit"]
        )
        df_nu["fnu_with_bg"] = df_nu["fnu"] + fnu_bg.value
    
        t_min = curve.build_axisarray(
            df_nu,
            "t",
            metadata_calc["t_unit"],
            conf["t_unit"]
        )
        fnu_mjy = curve.build_axisarray(
            df_nu,
            "fnu_with_bg",
            metadata_calc["fnu_unit"],
            conf["fnu_unit"]
        )
        curveconf = plot_utils.CurveConfigure(**conf["calculation"])
        curveconf.label = f"{nu_obs} {conf['nu_unit']}"

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
                np.asarray(df_obs["fnu"]),
                np.asarray(df_obs["t_err"]),
                np.asarray(df_obs["fnu_err"]),
                scatterconf
            )
            legend_handles = []
            # plot_utils.hlines(ax,conf,metadata_obs,legend_handles)
            plot_utils.annotation(ax,annot)
            ax.legend()

            pdf.savefig(fig)
            plt.close(fig)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "calculation_lc",
        type=Path,
    )
    parser.add_argument(
        "--observation_lc",
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


