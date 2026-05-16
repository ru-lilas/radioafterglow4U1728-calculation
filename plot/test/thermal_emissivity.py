
import matplotlib.pyplot as plt
from module.utilities import filereaders as fr
from pathlib import Path

def plot_main(df,ax):
    ax.loglog(
        df["x_m"],
        df["jnu"],
        color = "#000000",
        ls="-",
    )
    labelfs=16
    ax.set_xlabel(r"$x_{\mathrm{M}}$",fontsize=labelfs)
    ax.set_ylabel(r"$j_{\nu}$",fontsize=labelfs)


dfs = [
        fr.read_csv(Path("data/test/thermal_emissivity_theta002.csv")),
        fr.read_csv(Path("data/test/thermal_emissivity_theta010.csv")),
        fr.read_csv(Path("data/test/thermal_emissivity_theta050.csv"))
]

fig,ax = plt.subplots()
fig.set_layout_engine("constrained")
ax.loglog(
    dfs[0]["x_m"],
    dfs[0]["jnu"],
    color = "#FF4B00",
    ls="-",
    label=r"$\Theta=0.2$"
)
ax.loglog(
    dfs[1]["x_m"],
    dfs[1]["jnu"],
    color = "#000000",
    ls="-",
    label=r"$\Theta=1.0$"
)
ax.loglog(
    dfs[2]["x_m"],
    dfs[2]["jnu"],
    color = "#005AFF",
    ls="-",
    label=r"$\Theta=5.0$"
)
labelfs=16
ax.set_xlim(1.0e-3,1.0e+05)
ax.set_ylim(1.0e-6,1.0e+03)
ax.set_xlabel(r"$x_{\mathrm{M}}$",fontsize=labelfs)
ax.set_ylabel(r"$j_{\nu}~/~j_0$",fontsize=labelfs)
ax.legend()
fig.savefig("fig/test/thermal_emissivity.pdf")
