from module import plot_utils
from module import dataframe_utils
from module.strenums import KeyNames
from module.utilities import filereaders as fr
from pathlib import Path
import argparse
from module.strenums import PlotConfigNames,KeyNames
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument(
    "parameter_table",
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

parameter_path:Path = args.parameter_table

confpath:Path = args.config
outpath:Path = args.output
outpath.parent.mkdir(parents=True,exist_ok=True)

conf = fr.read_yaml(confpath)
df = fr.read_csv_within_idx(parameter_path)
metadata = fr.read_keyvalue(parameter_path)

with PdfPages(outpath) as pdf:
    fig,ax = plt.subplots(figsize=conf[PlotConfigNames.FIGSIZE])
    fig.set_layout_engine("constrained")

    xaxis_conf = conf["x"]
    x_ax = plot_utils.AxisConfigure(
        ax = ax,
        direction="x",
        lim = xaxis_conf["lim"],
        scale = xaxis_conf["scale"],
        ticksize = xaxis_conf["ticksize"],
        data = plot_utils.AxisArrayData(**xaxis_conf["data"]),
        label = plot_utils.LabelConfigure(**xaxis_conf["label"])
    )

    x = x_ax.array(df)

    yaxis_conf = conf["y"]
    y_ax = plot_utils.AxisConfigure(
        ax = ax,
        direction="y",
        lim = yaxis_conf["lim"],
        scale = yaxis_conf["scale"],
        ticksize = yaxis_conf["ticksize"],
        data = plot_utils.AxisArrayData(**yaxis_conf["data"]),
        label = plot_utils.LabelConfigure(**yaxis_conf["label"])
    )
    y = y_ax.array(df)

    cmconf = conf["colormap"]
    zconf = cmconf["z"]
    zname = str(zconf["name"])

    levels = np.concatenate([
        np.logspace(-2, 20, 64)
    ])

    z = dataframe_utils.extract_column_as_ndarray(df,zname)
    cf = ax.tricontourf(
        x, y, z,
        cmap = "cividis_r",
        norm=LogNorm(),
    )

    # colorbar
    cbar = fig.colorbar(cf, ax=ax)
    cbarconf = plot_utils.fetch_colorbar(
        conf,cbar
    )
    cbarconf.show()
    pdf.savefig(fig)
    plt.close(fig)

