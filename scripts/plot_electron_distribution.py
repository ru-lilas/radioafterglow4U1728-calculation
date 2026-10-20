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

ParameterName = Literal[
    "theta",
    "f_e",
    "p_index"
]

@dataclass(frozen=True,slots=True)
class DistributionParameters:
    theta: float
    f_e: float
    p_index: float
    gamma_e: NDArray[np.float64]

    def build_distribution(
        self,
    )->ElectronDistribution:
        return ElectronDistribution.compute(
            theta=self.theta,
            gamma=self.gamma_e,
            p_index=self.p_index,
            f_e=self.f_e
        )

@dataclass(frozen=True,slots=True)
class SweepSpec:
    parameter: ParameterName
    values: tuple[float,...]
    reference: DistributionParameters

    def build_params(
        self,
    )->list[DistributionParameters]:
        return [
            replace(
                self.reference,
                **{self.parameter:value}
            )
            for value in self.values
        ]

    def build_distributions(self)->list[ElectronDistribution]:
        return [
            params.build_distribution()
            for params in self.build_params()
        ]

reference = DistributionParameters(
    theta=1.0,
    f_e=1.0e-02,
    p_index=2.01,
    gamma_e=GAMMA_SPACE.geomspace()
)
theta_sweep = SweepSpec(
    parameter="theta",
    values=(
        0.5,
        1.0,
        5.0
    ),
    reference=reference
)
fe_sweep = SweepSpec(
    parameter="f_e",
    values=(
        1.0e-03,
        1.0e-02,
        1.0e-01
    ),
    reference=reference
)
p_sweep = SweepSpec(
    parameter="p_index",
    values=(
        2.01,
        2.50,
        3.00
    ),
    reference=reference
)

def main() -> None:
    args = parse_args()
    outpath: Path = args.output
    outpath.parent.mkdir(parents=True, exist_ok=True)

    plotconf = pu.PlotConfig.from_yaml(args.plotconfig)
    layout = plotconf.layout
    components = plotconf.components
    metadata_config = plotconf.metadata

    gamma_e = GAMMA_SPACE.geomspace()
    sweeps = (
        theta_sweep,
        p_sweep,
        fe_sweep
    )
    dists = [
        sweep.build_distributions()
        for sweep in sweeps
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
