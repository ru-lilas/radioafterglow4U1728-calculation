import matplotlib.pyplot as plt
from module.utilities import filereaders as fr
from module.utilities import plot_utils
from pathlib import Path

input_path = Path("data/test/thermal_spectrum_000.csv")
df = fr.read_csv(input_path)
metadata = fr.read_keyvalue(input_path)

def create_plot(figsize:tuple[float,float]):
    fig,ax = plt.subplots(figsize=figsize)
    fig.set_layout_engine("constrained")
    return fig,ax
def save_plot(fig,spectrum:str):
    fig.savefig(f"fig/test/thermal_{spectrum}.pdf")
    fig.savefig(f"fig/test/thermal_{spectrum}.svg")
    return

fig,ax = create_plot((8,6))
ax.loglog(
    df["nu"],
    df["jnu_th"],
    color = "#000000",
    ls="-",
)
labelfs=16
ax.set_xlim(1.0e+6,1.0e+15)
ax.set_ylim(1.0e-16,1.0e-10)
nu_crit = metadata["nu_crit_value"]
nu_B = metadata["nu_B_value"]
ax.axvline(nu_crit*1.0e-02,ls="--",color="#000000",lw=1.5)
ax.axvline(nu_crit,ls="-",color="#000000",lw=1.5)
ax.axvline(nu_crit*1.0e+02,ls="-.",color="#000000",lw=1.5)
ax.axvline(nu_B,ls="-",color="#004AFF",lw=1.0)
annotconf = plot_utils.AnnotationConfigure(
    use=True,
    fontsize=8,
    text= \
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
j_unit = metadata["j0_unit"]
nu_unit = metadata["nu_array_unit"]
ax.set_ylabel(r"$j_{\nu}~$"f"[{j_unit}]",fontsize=labelfs)
ax.set_xlabel(r"$\nu~$"f"[{nu_unit}]",fontsize=labelfs)
plot_utils.annotation(ax,annotconf)
save_plot(fig,"emissivity")
