
import matplotlib.pyplot as plt
from module.utilities import filereaders as fr
from pathlib import Path

dfs = [
        fr.read_csv(Path("data/test/thermal_spectrum_000.csv")),
]

def plot_spectrum_each_theta(ax,dfs,y:str,ylabel:str):
    ax.loglog(
        dfs[0]["x_m"],
        dfs[0][y],
        color = "#FF4B00",
        ls="-",
    )
    # ax.loglog(
    #     dfs[1]["x_m"],
    #     dfs[1][y],
    #     color = "#000000",
    #     ls="-",
    #     label=r"$\Theta=1.0$"
    # )
    # ax.loglog(
    #     dfs[2]["x_m"],
    #     dfs[2][y],
    #     color = "#005AFF",
    #     ls="-",
    #     label=r"$\Theta=5.0$"
    # )
    labelfs=16
    ax.set_xlim(1.0e-3,1.0e+05)
    ax.set_ylim(1.0e+10,1.0e+30)
    ax.set_ylabel(ylabel,fontsize=labelfs)
    ax.set_xlabel(r"$x_{\mathrm{M}}$",fontsize=labelfs)
    ax.legend()

def create_plot(figsize:tuple[float,float]):
    fig,ax = plt.subplots(figsize=figsize)
    fig.set_layout_engine("constrained")
    return fig,ax
def save_plot(fig,spectrum:str):
    fig.savefig(f"fig/test/thermal_{spectrum}.pdf")
    fig.savefig(f"fig/test/thermal_{spectrum}.svg")
    return

fig,ax = create_plot((8,6))
plot_spectrum_each_theta(ax,dfs,"lnu",r"$L_{\nu}$ [erg/s/Hz]")
save_plot(fig,"luminosity")
