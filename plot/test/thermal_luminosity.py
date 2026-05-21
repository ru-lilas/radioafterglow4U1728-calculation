
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from module.utilities import filereaders as fr
from module.utilities import plot_utils
from pathlib import Path
import numpy as np

input_path = Path("data/test/thermal_spectrum_000.csv")
df = fr.read_csv(input_path)
metadata = fr.read_keyvalue(input_path)

x_value = r"$\nu$"
x_unit = metadata["nu_array_unit"]
y_unit = "erg/s/Hz"
y_value = r"$L_{\nu}$"

conf_ticks = plot_utils.TicksConfigure(
    xlim=(1.0e+06,1.0e+15),
    ylim=(1.0e+10,1.0e+30),
    xscale="log",
    yscale="log",
    fontsize=16
)
conf_label = plot_utils.LabelConfigure(
    xlabel=x_value+f" [{x_unit}]",
    ylabel=y_value+f" [{y_unit}]",
    fontsize=16
)

def create_plot(figsize:tuple[float,float]):
    fig,ax = plt.subplots(figsize=figsize)
    fig.set_layout_engine("constrained")
    return fig,ax
def save_plot(fig,spectrum:str):
    fig.savefig(f"fig/test/thermal_{spectrum}.pdf")
    fig.savefig(f"fig/test/thermal_{spectrum}.svg")
    return

nu_array = df["nu"]
lnu_array = df["lnu"]
fig,ax = create_plot((8,6))
ax.loglog(
    nu_array,
    lnu_array,
    color = "#000000",
    ls="-",
)
plot_utils.configure_tick(ax,conf_ticks)
plot_utils.configure_label(ax,conf_label)

nu_crit = metadata["nu_crit_value"]
nu_crit_left = nu_crit*1.0e-02
nu_B = metadata["nu_B_value"]

# Rayleigh-Jeans limit
def extract_luminosity_for_given_nu(nu:float):
    idx = np.abs(np.log(df["nu"] / nu)).idxmin()
    nu_value = df.loc[idx,"nu"]
    lnu_value = df.loc[idx,"lnu"]
    return (nu_value,lnu_value)

def create_powerlaw_data(x1,y1,x2,y2):
    pl_index = np.log(y2/y1) / np.log(x2/x1)
    c = y1 / x1**pl_index
    
    x = np.logspace(3,15, 100)
    
    y = c * x**pl_index
    return {
        "alpha":pl_index,
        "c":c,
        "x":x,
        "y":y
    }

ref_rj = extract_luminosity_for_given_nu(nu_crit_left)
x_rj = np.logspace(3,15, 100)
y_rj = ref_rj[1] * (x_rj/nu_crit_left)**2

ax.loglog(x_rj,y_rj,label=r"$\propto\nu^2$",color="#84919E",ls="--",zorder=-2)

ax.axvline(nu_crit,ls="--",color="#000000",lw=1.5)
annotconf = plot_utils.AnnotationConfigure(
    use=True,
    fontsize=8,
    text=\
        r"$\beta_{\mathrm{sh}}=$"f"{metadata['beta_sh']:.1e}"
        f", $A=${metadata['a_wind_value']:.1e} {metadata['a_wind_unit']}"
        r", $\varepsilon_{\mathrm{th}}=$"f"{metadata['eps_th']:.1e}"
        r", $\varepsilon_{\mathrm{B}}=$"f"{metadata['eps_B']:.1e}"
        r", $\mu=$"f"{metadata['mu']:.2f}"
        r", $\mu_{\mathrm{e}}=$"f"{metadata['mu_e']:.2f}\n"
        f"$t=${metadata['t_value']:.1e} {metadata['t_unit']}"
        f", $r=${metadata['r_value']:.1e} {metadata['r_unit']}"
        r", $n_{\mathrm{w}}=$"f"{metadata['n_wind_value']:.1e} {metadata['n_wind_unit']}"
        r", $\Theta=$"f"{metadata['theta_e']:.1e}"
        f", $B=${metadata['b_mag_value']:.1e} {metadata['b_mag_unit']}"
)
legend_handles = [
    Line2D([0],[0], color="#000000", ls="-", label=r"$L_{\nu}$"),
    Line2D([0],[0], color="#84919E", ls="--", label=r"$L_{\nu}\propto\nu^{1/3}$"),
    Line2D([0],[0], color="#000000", ls="--", label=r"$\nu_{\mathrm{crit}}=$"f"{nu_crit:.1e} {x_unit}")
]
ax.legend(handles=legend_handles)
plot_utils.annotation(ax,annotconf)
save_plot(fig,"luminosity")
