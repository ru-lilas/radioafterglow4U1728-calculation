from module import dataframe_processors, dataframe_utils
from module.utilities import filereaders as fr
from pathlib import Path
import argparse
from module.plot import curve
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from module.plot import plot_utils
import numpy as np
import pandas as pd
from module.compute_lightcurve import BinnedCalculationLightcurve, CalculationLightcurve

def main(args:argparse.Namespace):
    path_lc:Path = args.lightcurve_data
    confpath:Path = args.config
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    lc = BinnedCalculationLightcurve.from_csv(path_lc)

    with PdfPages(outpath) as pdf:
        t_center = lc.t_center
        t_left = lc.t_left
        t_right = lc.t_right
        bin_edges = lc.bin_edges
        fnu = lc.fnu_averaged
    #     for nu_obs, df_obs in df_obs_raw.groupby(KeyNames.NU,sort=False):
    #         df_obs = df_obs.reset_index(drop=True)
    #         nu_obs = cast(float,nu_obs)
    #         df_nu = pd.DataFrame(df_calc_raw[np.isclose(df_calc_raw[KeyNames.NU], nu_obs)].reset_index(drop=True))
    #
    #         fnu_bg_value = np.asarray(df_obs["bg"],dtype=np.float64)[0]
    #
    #         fnu_bg = quantity_data.QuantityData(
    #             value=fnu_bg_value,
    #             unit=metadata_obs[KeyNames.FNU_UNIT]
    #         )
    #         fnu_bg_err = dataframe_utils.extract_column_as_ndarray(df_obs,LightcurveColumns.BG_ERR)[0]
    #         df_nu[KeyNames.FNU_WITH_BG] = df_nu[KeyNames.FNU_NET] + fnu_bg.value
    #
    #         t_min,fnu_mjy = build_chevalier_xy(
    #             df_nu,metadata_calc,conf
    #         )
    #
    #         curveconf = plot_utils.CurveConfigure(**conf["calculation"])
    #         curveconf.label = f"{nu_obs} {conf[KeyNames.NU_UNIT]}"
    #
    #         scatterconf = plot_scatter.ScatterConfigure(**conf["observation_scatter"])
    #         text = ""
    #         for key,element in conf["annotation"]["elements"].items():
    #             value = metadata_calc[key]
    #             prefix = element["prefix"]
    #             fmt = element["fmt"]
    #             text += f"{prefix}={value:{fmt}} "
    #
    #         for key,element in conf["annotation"]["from_tabledata"].items():
    #             prefix = element["prefix"]
    #             fmt = element["fmt"]
    #             value_arr = dataframe_processors.convert_ndarray(df_nu,key)
    #             # if element["tendency"] == "unique":
    #             value = float(value_arr[0])
    #             if element["with_unit"]:
    #                 unit = metadata_calc[f"{key}_unit"]
    #                 text += f"{prefix}={value:{fmt}} {unit} "
    #             else:
    #                 text += f"{prefix}={value:{fmt}} "
    #
    #
    #         annot = plot_utils.AnnotationConfigure(
    #             **conf["annotation"]["config"],
    #             text=text
    #         )
    #
        figsize=(16,12)
        fig,ax = plt.subplots(figsize=figsize)
        fig.set_layout_engine("constrained")

        ax.stairs(
            fnu,
            bin_edges,
        )
    #         curve.curve(ax,t_min,fnu_mjy,curveconf)
    #         plot_scatter.with_errorbar(
    #             ax,
    #             np.asarray(df_obs[KeyNames.T]),
    #             np.asarray(df_obs[KeyNames.FNU]),
    #             # np.asarray(df_obs[KeyNames.T_ERR]),
    #             None,
    #             np.asarray(df_obs[KeyNames.FNU_ERR]),
    #             scatterconf
    #         )
    #         legend_handles = []
    #         ax.axhline(fnu_bg_value,ls="--",color="#000000")
    #         plot_utils.annotation(ax,annot)
    #         ax.legend()
    #
        pdf.savefig(fig)
        plt.close(fig)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--lightcurve_data",
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
