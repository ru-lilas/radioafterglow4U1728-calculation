from module import plot_utils
from module.utilities import filereaders as fr
from pathlib import Path
import argparse
from module.strenums import PlotConfigNames
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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
        data = plot_utils.PlotData(**xaxis_conf["data"]),
        label = plot_utils.LabelConfigure(**xaxis_conf["label"])
    )

    yaxis_conf = conf["y"]
    y_ax = plot_utils.AxisConfigure(
        ax = ax,
        direction="y",
        lim = yaxis_conf["lim"],
        scale = yaxis_conf["scale"],
        ticksize = yaxis_conf["ticksize"],
        data = plot_utils.PlotData(**yaxis_conf["data"]),
        label = plot_utils.LabelConfigure(**yaxis_conf["label"])
    )

    contourconf = conf["contourf"]
    z_data = plot_utils.PlotData(**contourconf["zdata"])

    cmconf = conf["colormap"]

    levelconf = plot_utils.ContourLevelConfigure(
        **contourconf["levels"]
    )
    levels = levelconf.levels

    cmapconf = plot_utils.ColormapConfigure(**cmconf)
    cmap = cmapconf.build_cmap()

    meshgrid_data = plot_utils.build_meshgrid(
        df,
        x_column=x_ax.data.value,
        y_column=y_ax.data.value,
        z_column=z_data.value
    )

    cf = ax.contourf(
        *meshgrid_data,
        cmap = cmap,
        norm=LogNorm(vmin=levels[0], vmax=levels[-1]),
        extend="min",
        levels = levels
    )

    # colorbar
    cbar = fig.colorbar(cf, ax=ax)
    cbarconf = plot_utils.fetch_colorbar(
        conf,cbar
    )
    cbarconf.show()
    pdf.savefig(fig)
    plt.close(fig)
