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

parser = argparse.ArgumentParser()

parser.add_argument(
    "chi2_colormap",
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

cmap_path:Path = args.chi2_colormap

confpath:Path = args.config
outpath:Path = args.output
outpath.parent.mkdir(parents=True,exist_ok=True)

conf = fr.read_yaml(confpath)
df_param = fr.read_csv(cmap_path)
metadata_chi2 = fr.read_keyvalue(cmap_path)

df_set = dataframe_utils.build_dfs_grouped(
    df_param,group_by=KeyNames.NU
)

with PdfPages(outpath) as pdf:
    for nu,df in df_set:

        fig,ax = plt.subplots(figsize=conf[PlotConfigNames.FIGSIZE])
        fig.set_layout_engine("constrained")

        ax.set_title(f"{nu:.1f} {metadata_chi2[KeyNames.NU_UNIT]}")

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

        # levels
        contourconf = conf["contourf"]
        levelconf = plot_utils.ContourLevelConfigure(
            **contourconf["levels"]
        )
        levels = levelconf.levels

        # colorscale
        cmconf = conf["colormap"]
        cmapconf = plot_utils.ColormapConfigure(**cmconf)
        cmap = cmapconf.build_cmap()
        
        contourconf = conf["contourf"]
        z_data = plot_utils.PlotData(**contourconf["zdata"])
        meshgrid_data = plot_utils.build_meshgrid(
            df,
            x_column=x_ax.data.value,
            y_column=y_ax.data.value,
            z_column=z_data.value
        )

        cf = ax.contourf(
            *meshgrid_data,
            cmap=cmap,
            norm=LogNorm(vmin=levels[0], vmax=levels[-1]),
            extend="max",
            levels=levels
        )

        # colorbar
        cbar = fig.colorbar(cf, ax=ax)
        cbarconf = plot_utils.fetch_colorbar(
            conf,cbar
        )
        cbarconf.show()
        pdf.savefig(fig)
        plt.close(fig)
