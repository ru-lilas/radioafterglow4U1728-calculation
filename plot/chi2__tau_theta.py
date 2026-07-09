import matplotlib.pyplot as plt
from module import dataframe_utils
from module.utilities import filereaders as fr
from pathlib import Path
import argparse
from matplotlib.backends.backend_pdf import PdfPages
from module.strenums import KeyNames, PlotConfigNames
from module.plot import plot_utils

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

inpath:Path = args.input
confpath:Path = args.config
outpath:Path = args.output
outpath.parent.mkdir(parents=True,exist_ok=True)

conf = fr.read_yaml(confpath)
df = fr.read_csv(inpath)
metadata = fr.read_keyvalue(inpath)

conftick = plot_utils.TicksConfigure(**conf["TicksConfigure"])
conflabel = plot_utils.LabelConfigure(**conf["LabelConfigure"])

df_set = dataframe_utils.build_dfs_grouped(
    df,KeyNames.NU
)

with PdfPages(outpath) as pdf:
    for (nu, df_nu) in df_set:
        fig,ax = plt.subplots(figsize=conf[PlotConfigNames.FIGSIZE])
        fig.set_layout_engine("constrained")
        plot_utils.configure_tick(ax,conftick)
        plot_utils.configure_label(ax,conflabel)

        x = dataframe_utils.extract_column_as_ndarray(
            df_nu,KeyNames.TAU_THETA
        )
        y = dataframe_utils.extract_column_as_ndarray(
            df_nu,KeyNames.CHI2
        )

        ax.scatter(
            x,y,
            color = "#000000",
            ls = "-"
        )

        pdf.savefig(fig)
        plt.close(fig)
