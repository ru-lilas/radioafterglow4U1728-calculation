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
from module.electron_distributions import ElectronDistribution, MaxwellJuttnerDistribution, MaxwellJuttnerFactor, PowerlawDistribution, PowerlawNormalization

THETA_SPACE =  FloatVectorSpace(
    start=1.00e-01,
    stop =1.0e+02,
    num = 1024,
    scale="log"
)
THETA = THETA_SPACE.geomspace()

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
    "p_index"
]

PARAMETER_LABELS: dict[ParameterName, pu.QuantityLabel] = {
    "theta": pu.QuantityLabel(
        prefix=r"\Theta",
        fmt=".1f",
        unit="",
    ),
    "p_index": pu.QuantityLabel(
        prefix=r"g(\Theta),\,p",
        fmt=".2f",
        unit="",
    ),
}

@dataclass(frozen=True,slots=True)
class Parameters:
    p_index: float

    def build(
        self,
        theta_arr: NDArray[np.float64]
    ):
        f_factor = MaxwellJuttnerFactor.compute(theta_arr)

        gamma_min = np.asarray([
            MaxwellJuttnerDistribution.calculate_mean_gamma(theta)
            for theta in theta_arr
        ])

        g_factor = PowerlawNormalization.compute(
            p = self.p_index,
            theta = THETA,
            gamma_min=gamma_min
        )


@dataclass(frozen=True,slots=True)
class SweepSpec:
    parameter: ParameterName
    values: tuple[float,...]
    reference: Parameters

    def build_params(
        self,
    )->list[Parameters]:
        return [
            replace(
                self.reference,
                **{self.parameter:value}
            )
            for value in self.values
        ]

reference = Parameters(
    p_index=2.01,
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
sweeps = (p_sweep,)

def main() -> None:
    args = parse_args()
    outpath: Path = args.output
    outpath.parent.mkdir(parents=True, exist_ok=True)

    plotconf = pu.PlotConfig.from_yaml(args.plotconfig)
    layout = plotconf.layout
    components = plotconf.components
    metadata_config = plotconf.metadata

    gamma_e = THETA_SPACE.geomspace()

    style_f = pu.LineStyleConfig(
        colors=[
            "#000000",
        ],
        linestyle="-",
        linewidth=2.0
    )
    style_f_limit = replace(
        style_f,
        linestyle=":"
    )
    style_g = replace(
        style_f,
        colors=[
            "#005AFF",
            "#FF4B00",
            "#03AF7A"
        ]
    )
    style_g_limit = replace(
        style_g,
        linestyle="--"
    )
    components = {
        "f": pu.ComponentConfig(
            type="line",
            style=style_f,
        )
    }

    with PdfPages(outpath) as pdf:
        fig, axes = layout.create()
        for (row,col), ax in np.ndenumerate(axes):
            axis_config = plotconf.layout.axes.for_subplot(row,col)
            sweep = sweeps[int(col)]
            params = sweep.build_params()
            f_factor = MaxwellJuttnerFactor.compute(THETA)
            gamma_min = np.asarray([
                MaxwellJuttnerDistribution.calculate_mean_gamma(theta)
                for theta in THETA
            ])
            gNs = [
                PowerlawNormalization.compute(
                    p = param.p_index,
                    theta = THETA,
                    gamma_min=gamma_min
                )
                for param in params
            ]
            label = PARAMETER_LABELS[sweep.parameter]
            components["g_factor"] = pu.ComponentConfig(
                type="line",
                style=style_g,
                show_legend=True,
                label_format=label
            )

            legends = []
            plotter = pu.Plotter(ax)
            plotter.line(
                x=THETA,
                y=f_factor.exact,
                style=style_f.resolve(0)
            )
            plotter.line(
                x=THETA,
                y=np.full_like(
                    THETA,
                    1.0,
                    dtype=np.float64
                ),
                style=style_f_limit.resolve(0)
            )
            for i,gN in enumerate(gNs):
                param = params[i]
                p_index = param.p_index
                line=plotter.line(
                    x=THETA,
                    y=gN.exact,
                    style=style_g.resolve(i)
                )
                plotter.line(
                    x=THETA,
                    y=np.full_like(
                        THETA,
                        (p_index-1.0)/(p_index-2.0),
                        dtype=np.float64
                    ),
                    style=style_g_limit.resolve(i)
                )
                legends.append(
                    pu.LegendEntry(
                        components["g_factor"],
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
        pdf.savefig(fig)
        plt.close(fig)

if __name__ == "__main__":
    main()
