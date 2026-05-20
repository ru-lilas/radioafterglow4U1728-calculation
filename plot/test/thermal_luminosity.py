
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
    df["lnu"],
    color = "#000000",
    ls="-",
)
labelfs=16
ax.set_xlim(1.0e+3,1.0e+12)
ax.set_ylim(1.0e+10,1.0e+30)
ax.set_ylabel(r"$L_{\nu}$ [erg/s/Hz]",fontsize=labelfs)
ax.set_xlabel(r"$\nu$ [Hz]",fontsize=labelfs)
annotconf = plot_utils.AnnotationConfigure(
    use=True,
    fontsize=12,
    text=r"$\Theta=$"f"{metadata['theta_e']:.1e}"
        f", $B=${metadata['b_mag_value']:.1e} {metadata['b_mag_unit']}"
        f", $A=${metadata['a_wind_value']:.1e} {metadata['a_wind_unit']}"
        f", $r=${metadata['r_value']:.1e} {metadata['r_unit']}"
        f", $t=${metadata['t_value']:.1e} {metadata['t_unit']}"
)
plot_utils.annotation(ax,annotconf)
save_plot(fig,"luminosity")
