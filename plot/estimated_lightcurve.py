from module.utilities import filereaders as fr
from pathlib import Path
import argparse
from module.plot import curve
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from module.plot import plot_utils
import numpy as np
from module import quantity
from module.plot import plot_scatter
from module import dataframe_processors as dfp
import pandas as pd

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
    
    t_sec = quantity.QuantityData(
        value = dfp.convert_ndarray(df_nu,"t_value"),
        unit = metadata_calc["t_unit"]
    )
    t_min = t_sec.unit_convert(conf["t_unit"])
    fnu_mjy = dfp.convert_ndarray(df_nu,"fnu_value")
    curveconf = plot_utils.CurveConfigure(**conf["calculation"])
    curveconf.label = f"{nu_values[1]} {conf['nu_unit']}"

    scatterconf = plot_scatter.ScatterConfigure(**conf["observation_scatter"])

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
        # calculation lightcurve
        curve.curve(ax,t_min.value,fnu_mjy,curveconf)
        plot_scatter.scatter_only(
            ax,
            np.asarray(df_obs["t"]),
            np.asarray(df_obs["f9_net"]),
            scatterconf
        )
        # annot = plot_utils.AnnotationConfigure(
        #     use=True,
        #     fontsize=16,
        #     text=\
        #         r"$\varepsilon_{B}=$"f"{metadata_contour['eps_B']}"
        #         r", $\varepsilon_\mathrm{th}=$"f"{metadata_contour['eps_th']}"
        #         r", $\mu=$"f"{metadata_contour['mu']}"
        #         r", $\mu_e=$"f"{metadata_contour['mu_e']}"
        # )
        # plot_utils.annotation(ax,annot)
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


