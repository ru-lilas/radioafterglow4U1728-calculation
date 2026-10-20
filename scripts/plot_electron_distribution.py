import argparse
from pathlib import Path
from typing import Hashable, Literal, get_args
import numpy as np

import matplotlib.pyplot as plt
from numpy.typing import NDArray
import pandas as pd
from dataclasses import fields, dataclass, replace
from matplotlib.backends.backend_pdf import PdfPages

from module import plot_utils as pu
from module.models import ColumnNames, FluxName, MetadataNames, ModelParameters, SwitchKeys
from module.mydataclasses import FloatVectorSpace, LongformatDataFrame, QuantityArray1D, QuantityData
from module.observation import LongformatLightcurve
from module.electron_distributions import ElectronDistribution, MaxwellJuttnerDistribution, PowerlawDistribution

GAMMA_SPACE =  FloatVectorSpace(
    start=1.01e+00,
    stop =1.0e+03,
    num = 2048,
    scale="log"
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "plotconfig",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args()

# @dataclass(frozen=True,slots=True)
# class ElectronDistributionParameterDependence:

def calculate_thetasweep(gamma_e:NDArray[np.float64]):
    f_e = 1.0e-02
    p_index=2.01
    theta_sweep = [
        5.0e-01,
        1.0e+00,
        5.0e+00
    ]

    dists = [
        ElectronDistribution.compute(
            theta=theta,
            gamma=gamma_e,
            gamma_min=None,
            p_index=p_index,
            f_e=f_e
        )
        for theta in theta_sweep
    ]
    return dists

def calculate_fesweep(gamma_e:NDArray[np.float64]):
    fe_sweep = [
        1.0e-03,
        1.0e-02,
        1.0e-01
    ]
    p_index=2.01
    theta = 1.0

    dists = [
        ElectronDistribution.compute(
            theta=theta,
            gamma=gamma_e,
            gamma_min=None,
            p_index=p_index,
            f_e=f_e
        )
        for f_e in fe_sweep
    ]
    return dists

def calculate_psweep(gamma_e:NDArray[np.float64]):
    f_e = 1.0e-01
    p_indices=[
        2.01,
        2.50,
        3.00
    ]
    theta = 1.0

    dists = [
        ElectronDistribution.compute(
            theta=theta,
            gamma=gamma_e,
            gamma_min=None,
            p_index=p_index,
            f_e=f_e
        )
        for p_index in p_indices
    ]
    return dists

def main() -> None:
    args = parse_args()
    outpath: Path = args.output
    outpath.parent.mkdir(parents=True, exist_ok=True)

    plotconf = pu.PlotConfig.from_yaml(args.plotconfig)
    layout = plotconf.layout
    components = plotconf.components
    metadata_config = plotconf.metadata

    gamma_e = GAMMA_SPACE.geomspace()

    dists_ele = calculate_thetasweep(gamma_e)
    dist_fe = calculate_fesweep(gamma_e)
    dist_p = calculate_psweep(gamma_e)
    dists = [
        dists_ele,
        dist_p,
        dist_fe
    ]

    style_thermal = pu.LineStyleConfig(
            colors=[
                "#000000",
                "#005AFF",
                "#FF4B00"
            ],
            linestyle="--"
    )
    style_powerlaw = replace(
        style_thermal,
        linestyle=":"
    )
    style_total = replace(
        style_thermal,
        linestyle="-"
    )

    # totdist = ElectronDistribution.compute(
    #     theta=theta,
    #     gamma=gamma_e,
    #     gamma_min=None,
    #     p_index=p_index,
    #     f_e=f_e
    # )
    # totfrac = totdist.integrated_number_fraction()
    # metadata:dict[str,QuantityData] = {
    #     "f_e":QuantityData(f_e,""),
    #     "p_index":QuantityData(p_index,"")
    # }

    with PdfPages(outpath) as pdf:
        fig, axes = layout.create()
        for (row,col), ax in np.ndenumerate(axes):
            axis_config = plotconf.layout.axes.for_subplot(row,col)
            dists_ele = dists[int(col)]
            plotter = pu.Plotter(ax)
            legends = []
            for i,dist in enumerate(dists_ele):
                print(dist.powerlaw.f_e)
                plotter.line(
                    x=gamma_e,
                    y=dist.thermal.gamma_pdf,
                    style=style_thermal.resolve(i)
                )
                plotter.line(
                    x=gamma_e,
                    y=dist.powerlaw.gamma_pdf,
                    style=style_powerlaw.resolve(i)
                )
                plotter.line(
                    x=gamma_e,
                    y=dist.gamma_pdf,
                    style=style_total.resolve(i)
                )
    #             print(
    #                 f"{totdist.integrated_number_fraction()[i]:.2f}\t"
    #                 f"{totdist.integrated_kinetic_energy()[i]:.2f}"
    #             )
    #             legends.append(
    #                 plot_utils.LegendEntry(
    #                     components["total"],
    #                     line[0],
    #                     value=theta[i]
    #                 )
    #             )
    #         plotter.legend(
    #             entries= legends,
    #             config = components["legend"]
    #         )
    #         strings: list[str] = []
    #         for name, quantity in metadata.items():
    #             plot_utils.MetadataComponent(
    #                 quantity=quantity,
    #                 config=plotconf.metadata.items[name],
    #             ).add_strings(strings)
    #         plotter.title(
    #             strings=strings,
    #             fontsize=plotconf.metadata.fontsize,
    #         )
        pdf.savefig(fig)
        plt.close(fig)

if __name__ == "__main__":
    main()
