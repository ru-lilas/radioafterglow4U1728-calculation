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
from module import plot_utils

def main(args:argparse.Namespace):
    confpath:Path = args.config
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    lc = BinnedCalculationLightcurve.from_csv(
        args.lc
    )
    conf = plot_utils.PlotConfig.from_yaml(
        args.config
    )

    with PdfPages(outpath) as pdf:
        fig,ax = plot_utils.create_configured_axes(conf)
        fig.set_layout_engine("constrained")

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
        "--lc",
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
