
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from module.utilities import filereaders as fr
from module.utilities import plot_utils
from pathlib import Path
import numpy as np
import argparse

extensions = [".pdf",".svg"]

def main(args:argparse.Namespace):

    inpath:Path = args.infile
    fn = inpath.stem
    metadata, df = fetch_metadata_df(inpath)

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
    
    # Rayleigh-Jeans limit
    def extract_luminosity_for_given_nu(nu:float):
        idx = np.abs(np.log(df["nu"] / nu)).idxmin()
        nu_value = df.loc[idx,"nu"]
        lnu_value = df.loc[idx,"lnu"]
        return (nu_value,lnu_value)
    
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
    save_plot(fig,f"luminosity_{fn}")

    return

def create_plot(figsize:tuple[float,float]):
    fig,ax = plt.subplots(figsize=figsize)
    fig.set_layout_engine("constrained")
    return fig,ax

def save_plot(fig,fn:str):
    base_dir = Path("fig/thermal_only/beta")
    base_dir.mkdir(parents=True,exist_ok=True)
    for extension in extensions:
        outpath = Path(f"{base_dir}/{fn}{extension}")
        fig.savefig(outpath)
        print(f"output {outpath}")
    return

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
    
    args = parser.parse_args()
    
    main(args)
