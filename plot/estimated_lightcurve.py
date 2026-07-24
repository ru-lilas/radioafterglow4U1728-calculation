from typing import Any, cast
from module import dataframe_processors, dataframe_utils
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
from module.strenums import KeyNames, LightcurveColumns

def build_chevalier_xy(
    df_nu:pd.DataFrame,
    metadata_calc:dict[str,Any],
    conf:dict[str,Any]
):
    t_min = curve.build_axisarray(
        df_nu,
        KeyNames.T,
        metadata_calc[KeyNames.T_UNIT],
        conf[KeyNames.T_UNIT]
    )
    fnu_mjy = curve.build_axisarray(
        df_nu,
        KeyNames.FNU_WITH_BG,
        metadata_calc[KeyNames.FNU_UNIT],
        conf[KeyNames.FNU_UNIT]
    )
    return t_min, fnu_mjy

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

    with PdfPages(outpath) as pdf:
        for nu_obs, df_obs in df_obs_raw.groupby(KeyNames.NU,sort=False):
            df_obs = df_obs.reset_index(drop=True)
            nu_obs = cast(float,nu_obs)
            df_nu = pd.DataFrame(df_calc_raw[np.isclose(df_calc_raw[KeyNames.NU], nu_obs)].reset_index(drop=True))

            fnu_bg_value = np.asarray(df_obs["bg"],dtype=np.float64)[0]

            fnu_bg = quantity_data.QuantityData(
                value=fnu_bg_value,
                unit=metadata_obs[KeyNames.FNU_UNIT]
            )
            fnu_bg_err = dataframe_utils.extract_column_as_ndarray(df_obs,LightcurveColumns.BG_ERR)[0]
            df_nu[KeyNames.FNU_WITH_BG] = df_nu[KeyNames.FNU_NET] + fnu_bg.value

            t_min,fnu_mjy = build_chevalier_xy(
                df_nu,metadata_calc,conf
            )

            curveconf = plot_utils.CurveConfigure(**conf["calculation"])
            curveconf.label = f"{nu_obs} {conf[KeyNames.NU_UNIT]}"

            scatterconf = plot_scatter.ScatterConfigure(**conf["observation_scatter"])
            text = ""
            for key,element in conf["annotation"]["elements"].items():
                value = metadata_calc[key]
                prefix = element["prefix"]
                fmt = element["fmt"]
                text += f"{prefix}={value:{fmt}} "

            for key,element in conf["annotation"]["from_tabledata"].items():
                prefix = element["prefix"]
                fmt = element["fmt"]
                value_arr = dataframe_processors.convert_ndarray(df_nu,key)
                # if element["tendency"] == "unique":
                value = float(value_arr[0])
                if element["with_unit"]:
                    unit = metadata_calc[f"{key}_unit"]
                    text += f"{prefix}={value:{fmt}} {unit} "
                else:
                    text += f"{prefix}={value:{fmt}} "


            annot = plot_utils.AnnotationConfigure(
                **conf["annotation"]["config"],
                text=text
            )

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
                np.asarray(df_obs[KeyNames.T]),
                np.asarray(df_obs[KeyNames.FNU]),
                # np.asarray(df_obs[KeyNames.T_ERR]),
                None,
                np.asarray(df_obs[KeyNames.FNU_ERR]),
                scatterconf
            )
            legend_handles = []
            ax.axhline(fnu_bg_value,ls="--",color="#000000")
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
