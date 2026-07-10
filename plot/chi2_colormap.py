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
        print(df)

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
        z = dataframe_utils.extract_column_as_ndarray(df,zname)

        levels = np.concatenate([
            np.logspace(-1, 2, 61),
        ])

        cmap = plt.get_cmap("cividis_r").copy()
        cmap.set_over("lightgray")    # 10より大きい値はグレー

        
        norm = LogNorm(vmin=1e-1, vmax=1e2)

        pivot = df.pivot(
            index="beta_sh",
            columns="a_wind",
            values="chi2",
        )

        X, Y = np.meshgrid(
            pivot.columns.to_numpy(),
            pivot.index.to_numpy(),
        )

        Z = pivot.to_numpy()
        cf = ax.contourf(
            X, Y, Z,
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
