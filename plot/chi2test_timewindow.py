import matplotlib.pyplot as plt
from module import dataframe_utils
from module.input_reader import InputReader
from module.utilities import filereaders as fr
from pathlib import Path
import argparse
from matplotlib.backends.backend_pdf import PdfPages
from module.strenums import KeyNames, PlotConfigNames
from module.plot import plot_utils
from module.input_dataclasses import PhysicalParameters

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    type=Path,
)
parser.add_argument(
    "-c",
    "--config",
    type=Path,
    required=True
)
parser.add_argument(
    "--parameters",
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

inpath:Path = args.input
confpath:Path = args.config
path_params:Path = args.parameters
outpath:Path = args.output
outpath.parent.mkdir(parents=True,exist_ok=True)

conf = fr.read_yaml(confpath)
params = InputReader.read(path_params,PhysicalParameters)
df = fr.read_csv(inpath)
dfs = dataframe_utils.build_dfs_grouped(
    df,KeyNames.NU
)
conftick = plot_utils.TicksConfigure(**conf["TicksConfigure"])
conflabel = plot_utils.LabelConfigure(**conf["LabelConfigure"])

with PdfPages(outpath) as pdf:
    for (nu, df_nu) in dfs:
        fig,ax = plt.subplots(figsize=conf[PlotConfigNames.FIGSIZE])
        fig.set_layout_engine("constrained")
        plot_utils.configure_tick(ax,conftick)
        plot_utils.configure_label(ax,conflabel)

        mask = df_nu["reject"]

        x = dataframe_utils.extract_column_as_ndarray(
            df_nu,"t_max"
        )
        y = dataframe_utils.extract_column_as_ndarray(
            df_nu,"p_value"
        )
        ax.set_title(
            f"{nu:.1f} GHz,"
            r'$t_{\min}=0$ min,'
            r'$\varepsilon_{\mathrm{th}}=$'f"{params.microphysics.eps_th:.2f},"
            r'$\varepsilon_{\mathrm{B}}=$'f"{params.microphysics.eps_b:.1e}",
        )
        ax.axhline(0.05,ls="--")

        ax.scatter(
            df_nu.loc[~mask,"t_max"],
            df_nu.loc[~mask,"p_value"],
            color = "#000000",
            marker = "o"
        )
        ax.scatter(
            df_nu.loc[mask,"t_max"],
            df_nu.loc[mask,"p_value"],
            color = "#000000",
            marker = "x"
        )
        #
        # ax.scatter(
        #     x,y,
        #     color = "#000000",
        # )
        pdf.savefig(fig)
        plt.close(fig)
