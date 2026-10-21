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

PARAMETER_LABELS: dict[ParameterName, pu.QuantityLabel] = {
    "theta": pu.QuantityLabel(
        prefix=r"\Theta",
        fmt=".1f",
        unit="",
    ),
    "p_index": pu.QuantityLabel(
        prefix="p",
        fmt=".2f",
        unit="",
    ),
    "f_e": pu.QuantityLabel(
        prefix=r"f_{\mathrm{e}}",
        fmt=".3g",
        unit="",
    ),
}


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
    p_index=2.0001,
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
        2.0001,
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
    components = {
        "thermal": pu.ComponentConfig(
            type="line",
            style=style_thermal,
        ),
        "powerlaw": pu.ComponentConfig(
            type="line",
            style=style_powerlaw,
        ),
    }

    with PdfPages(outpath) as pdf:
        fig, axes = layout.create()
        for (row,col), ax in np.ndenumerate(axes):
            axis_config = plotconf.layout.axes.for_subplot(row,col)
            sweep = sweeps[int(col)]
            dists_ele = sweep.build_distributions()
            label = PARAMETER_LABELS[sweep.parameter]
            components["total"] = pu.ComponentConfig(
                type="line",
                style=style_total,
                show_legend=True,
                label_format=label
            )

            legends = []
            plotter = pu.Plotter(ax)
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
                line=plotter.line(
                    x=gamma_e,
                    y=dist.gamma_pdf,
                    style=style_total.resolve(i)
                )
                legends.append(
                    pu.LegendEntry(
                        components["total"],
                        line[0],
                        value=sweep.values[i]
                    )
                )
            plotter.legend(
                entries= legends,
                config = pu.ComponentConfig(
                    type="legend",
                    style=pu.LegendStyle(
                        fontsize=28,
                        loc="upper right"
                    )
                )
            )
            strings = [
                label.to_text(getattr(sweep.reference, name))
                for name, label in PARAMETER_LABELS.items()
                if name != sweep.parameter
            ]
            plotter.title(
                strings=strings,
                fontsize=plotconf.metadata.fontsize,
            )
        pdf.savefig(fig)
        plt.close(fig)

if __name__ == "__main__":
    main()
