
import matplotlib.pyplot as plt
from module.utilities import filereaders as fr
from module.utilities import plot_utils
from pathlib import Path
import argparse

extensions = [".pdf",".svg"]

def main(args:argparse.Namespace):

    inpath:Path = args.infile
    df = fr.read_csv(inpath)

    x_value = r"$(\beta\gamma)_{\mathrm{sh}}$"
    y_unit = "erg/s/Hz"
    y_value = r"$L_{\nu,\mathrm{peak}}$"
    
    conf_ticks = plot_utils.TicksConfigure(
        xlim=(0.1,0.5),
        ylim=(1.0e+20,1.0e+26),
        xscale="log",
        yscale="log",
        fontsize=16
    )
    conf_label = plot_utils.LabelConfigure(
        xlabel=x_value,
        ylabel=y_value+f" [{y_unit}]",
        fontsize=16
    )
    x_array = df["betagamma_sh"]
    y_array = df["lnu_peak"]
    fig,ax = create_plot((8,6))

    ax.scatter(
        x_array,y_array
    )
    plot_utils.configure_tick(ax,conf_ticks)
    plot_utils.configure_label(ax,conf_label)
    
    fig.savefig(args.outpath)
    print(f"output {args.outpath}")

    return

def create_plot(figsize:tuple[float,float]):
    fig,ax = plt.subplots(figsize=figsize)
    fig.set_layout_engine("constrained")
    return fig,ax

# def save_plot(fig,fn:str):
#     base_dir = Path("fig/thermal_only/beta")
#     base_dir.mkdir(parents=True,exist_ok=True)
#     for extension in extensions:
#         outpath = Path(f"{base_dir}/{fn}{extension}")
#         fig.savefig(outpath)
#         print(f"output {outpath}")
#     return

def fetch_metadata_df(inpath:Path):
    print(f"read {inpath}")
    df = fr.read_csv(inpath)
    metadata = fr.read_keyvalue(inpath)
    return metadata,df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "infile",
        type=Path
    )
    parser.add_argument(
        "outpath",
        type=Path
    )
    
    args = parser.parse_args()
    
    main(args)

