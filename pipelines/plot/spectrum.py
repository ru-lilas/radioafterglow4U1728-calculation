# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false

import matplotlib.pyplot as plt
from module.utilities import filereaders as fr
from module.utilities import plot_utils
from pathlib import Path
import argparse
import astropy.units as u

def main(args:argparse.Namespace):
    inpath:Path = args.input
    if not inpath.exists():
        print(f"エラー：存在しないパス {inpath}")
    confpath:Path = args.config
    outpath:Path = args.output

    conf = fr.read_yaml(confpath)
    conftick = plot_utils.TicksConfigure(**conf["TicksConfigure"])
    conflabel = plot_utils.LabelConfigure(**conf["LabelConfigure"])

    metadata = fr.read_keyvalue(inpath)
    df = fr.read_csv(inpath)

    x_name = conf["x_name"]
    y_name = conf["y_name"]

    figsize=(8,6)
    fig,ax = plt.subplots(figsize=figsize)
    fig.set_layout_engine("constrained")
    plot_utils.configure_tick(ax,conftick)
    plot_utils.configure_label(ax,conflabel)
    ax.loglog(
        df[x_name],
        df[y_name],
        color = "#000000",
        ls="-",
    )
    t_value = metadata["t_value"]
    t_unit = metadata["t_unit"]
    t = u.Quantity(t_value,u.Unit(t_unit))
    annotconf = plot_utils.AnnotationConfigure(
        use=True,
        fontsize=12,
        text = \
        r"$t=$"f"{t:.1e}"
    )
    # annotconf = plot_utils.AnnotationConfigure(
    #     use=True,
    #     fontsize=8,
    #     text=\
    #         r"$\beta_{\mathrm{sh}}=$"f"{metadata['beta_sh']:.1e}"
    #         f", $A=${metadata['a_wind_value']:.1e} {metadata['a_wind_unit']}"
    #         r", $\varepsilon_{\mathrm{th}}=$"f"{metadata['eps_th']:.1e}"
    #         r", $\varepsilon_{\mathrm{B}}=$"f"{metadata['eps_B']:.1e}"
    #         r", $\mu=$"f"{metadata['mu']:.2f}"
    #         r", $\mu_{\mathrm{e}}=$"f"{metadata['mu_e']:.2f}\n"
    #         f"$t=${metadata['t_value']:.1e} {metadata['t_unit']}"
    #         f", $r=${metadata['r_value']:.1e} {metadata['r_unit']}"
    #         r", $n_{\mathrm{w}}=$"f"{metadata['n_wind_value']:.1e} {metadata['n_wind_unit']}"
    #         r", $\Theta=$"f"{metadata['theta_e']:.1e}"
    #         f", $B=${metadata['b_mag_value']:.1e} {metadata['b_mag_unit']}"
    # )
    outpath.parent.mkdir(parents=True,exist_ok=True)
    plot_utils.annotation(ax,annotconf)
    fig.savefig(outpath)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "input",
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
    
    main(args)
