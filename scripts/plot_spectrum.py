import argparse
from pathlib import Path
from typing import Any, Hashable, Literal, get_args
import numpy as np

import matplotlib.pyplot as plt
from numpy.typing import NDArray
import pandas as pd
from dataclasses import fields, dataclass, replace
from matplotlib.backends.backend_pdf import PdfPages

from module import utils
from module.dataframe_utils import groupby_to_dict, select_maximum_row
from module.mydataclasses import FloatVectorSpace, LongformatDataFrame, QuantityArray1D, QuantityData
from module.utils import FileReader, FileWriter
from module.timing_utils import timed
from module.models import ModelParameters,ColumnNames,SynchrotronResults
from module.synchrotron_numerical_table import ThermalSynchrotronIntegrals
from module import plot_utils as pu
import astropy.units as u

XM_SPACE = FloatVectorSpace(
    start=1.0e-02,
    stop =1.0e+04,
    num = 1024,
    scale="log"
)
XM = XM_SPACE.geomspace()
XM = XM[None,:]

figure = pu.FigureConfig(
    figsize=(12*2,9*5)
)
subplots = pu.SubplotConfig(
    nrows=5,
    ncols=2,
    sharex="col",
    sharey="row",
    hide_shared_labels=True
)
x_axis = pu.AxisConfig(
    scale="log",
    limits=(1.0e+07,1.0e+12),
    label = pu.LabelConfigure(
        prefix="Frequency",
        unit="Hz",
        fontsize=32
    )
)
y_axis = replace(
    x_axis,
    limits=(1.0e-02,1.0e+06),
    label = pu.LabelConfigure(
        prefix="SSA Optical Depth",
        unit="",
        fontsize=32
    )
)
lnu_axis = replace(
    x_axis,
    limits=(1.0e+14,1.0e+20),
    label = pu.LabelConfigure(
        prefix=r'$L_{\nu}$',
        unit="erg s-1 Hz-1",
        fontsize=32
    )
)
fnu_axis = replace(
    x_axis,
    limits=(1.0e-05,1.0e+01),
    label = pu.LabelConfigure(
        prefix=r'$F_{\nu}$',
        unit="mJy",
        fontsize=32
    )
)
jnu_axis = replace(
    x_axis,
    limits=(1.0e-24,1.0e-12),
    label = pu.LabelConfigure(
        prefix=r'$j_{\nu}$',
        unit="erg cm-3 s-1 Hz-1",
        fontsize=32
    )
)
anu_axis = replace(
    x_axis,
    limits=(1.0e-16,1.0e+00),
    label = pu.LabelConfigure(
        prefix=r'$\alpha_{\nu}$',
        unit="cm-1",
        fontsize=32
    )
)
ticks = pu.TickStyle(
    labelsize=32,
    direction="in",
)
AXES = pu.AxesConfig(
    x=x_axis,
    y=(
        jnu_axis,
        anu_axis,
        y_axis,
        lnu_axis,
        fnu_axis
    ),
    ticks=ticks
)
layout = pu.PlotLayout(
    figure,AXES,subplots
)
STYLE_FIDUCIAL = pu.LineStyle(
    color="#000000",
    linestyle="-",
    linewidth=2.0
)
styles = [
    replace(STYLE_FIDUCIAL,color="#FF4B00",linestyle="-"),
    STYLE_FIDUCIAL,
    replace(STYLE_FIDUCIAL,color="#005FBB",linestyle="-")
]
from matplotlib.lines import Line2D

@dataclass(frozen=True,slots=True)
class PlotComponentBase:
    x: NDArray[np.float64]
    thermal: NDArray[np.float64]
    powerlaw: NDArray[np.float64]
    total: NDArray[np.float64]
    style: pu.LineStyle

    def draw(
        self,
        plotter:pu.Plotter,
        tau: NDArray[np.float64],
        style:pu.LineStyle
    ):
        return plotter.line(
            x=self.x,
            y=tau,
            style=style
        )

    def draw_thermal(
        self,
        plotter:pu.Plotter
    ):
        self.draw(
            plotter,self.thermal,
            replace(
                self.style,
                linestyle="--"
            )
        )
    def draw_powerlaw(
        self,
        plotter:pu.Plotter
    ):
        self.draw(
            plotter,self.powerlaw,
            replace(
                self.style,
                linestyle=":"
            )
        )
    def draw_total(
        self,
        plotter:pu.Plotter
    ):
        self.draw(
            plotter,self.total,
            replace(
                self.style,
                linestyle="-"
            )
        )

    def draw_all(
        self,
        plotter:pu.Plotter
    ):
        self.draw_powerlaw(plotter)
        self.draw_thermal(plotter)
        self.draw_total(plotter)

@dataclass(frozen=True,slots=True)
class PlotOpticalDepth(PlotComponentBase):

    @classmethod
    def from_df(
        cls,
        x:NDArray[np.float64],
        df:pd.DataFrame,
        style:pu.LineStyle
    ):
        return cls(
            x=x,
            thermal = df["tau_thermal"].to_numpy(dtype=np.float64),
            powerlaw = df["tau_powerlaw"].to_numpy(dtype=np.float64),
            total = df["tau_total"].to_numpy(dtype=np.float64),
            style=style
        )

@dataclass(frozen=True,slots=True)
class PlotSpecificLuminosity(PlotComponentBase):
    @classmethod
    def from_df(
        cls,
        x:NDArray[np.float64],
        df:pd.DataFrame,
        unit: str,
        style: pu.LineStyle
    ):
        df_unit = df.attrs["lnu_unit"]
        thermal = u.Quantity(
            df["lnu_thermal"].to_numpy(dtype=np.float64),
            df_unit
        )
        powerlaw = u.Quantity(
            df["lnu_powerlaw"].to_numpy(dtype=np.float64),
            df_unit
        )
        total = u.Quantity(
            df["lnu_total"].to_numpy(dtype=np.float64),
            df_unit
        )
        return cls(
            x=x,
            thermal = np.asarray(thermal.to_value(unit),dtype=np.float64),
            powerlaw = np.asarray(powerlaw.to_value(unit),dtype=np.float64),
            total = np.asarray(total.to_value(unit),dtype=np.float64),
            style=style
        )

@dataclass(frozen=True,slots=True)
class PlotSpecificFlux(PlotComponentBase):
    @classmethod
    def from_df(
        cls,
        x:NDArray[np.float64],
        df:pd.DataFrame,
        unit: str,
        style: pu.LineStyle
    ):
        df_unit = df.attrs["fnu_unit"]
        thermal = u.Quantity(
            df["fnu_thermal"].to_numpy(dtype=np.float64),
            df_unit
        )
        powerlaw = u.Quantity(
            df["fnu_powerlaw"].to_numpy(dtype=np.float64),
            df_unit
        )
        total = u.Quantity(
            df["fnu_total"].to_numpy(dtype=np.float64),
            df_unit
        )
        return cls(
            x=x,
            thermal = np.asarray(thermal.to_value(unit),dtype=np.float64),
            powerlaw = np.asarray(powerlaw.to_value(unit),dtype=np.float64),
            total = np.asarray(total.to_value(unit),dtype=np.float64),
            style=style
        )

@dataclass(frozen=True,slots=True)
class PlotEmissivity(PlotComponentBase):
    @classmethod
    def from_df(
        cls,
        x:NDArray[np.float64],
        df:pd.DataFrame,
        unit: str,
        style: pu.LineStyle
    ):
        df_unit = df.attrs["jnu_unit"]
        thermal = u.Quantity(
            df["jnu_thermal"].to_numpy(dtype=np.float64),
            df_unit
        )
        powerlaw = u.Quantity(
            df["jnu_powerlaw"].to_numpy(dtype=np.float64),
            df_unit
        )
        total = u.Quantity(
            df["jnu_total"].to_numpy(dtype=np.float64),
            df_unit
        )
        return cls(
            x=x,
            thermal = np.asarray(thermal.to_value(unit),dtype=np.float64),
            powerlaw = np.asarray(powerlaw.to_value(unit),dtype=np.float64),
            total = np.asarray(total.to_value(unit),dtype=np.float64),
            style=style
        )

@dataclass(frozen=True,slots=True)
class PlotAbsorption(PlotComponentBase):
    @classmethod
    def from_df(
        cls,
        x:NDArray[np.float64],
        df:pd.DataFrame,
        unit: str,
        style: pu.LineStyle
    ):
        df_unit = df.attrs["alpha_nu_unit"]
        thermal = u.Quantity(
            df["alpha_nu_thermal"].to_numpy(dtype=np.float64),
            df_unit
        )
        powerlaw = u.Quantity(
            df["alpha_nu_powerlaw"].to_numpy(dtype=np.float64),
            df_unit
        )
        total = u.Quantity(
            df["alpha_nu_total"].to_numpy(dtype=np.float64),
            df_unit
        )
        return cls(
            x=x,
            thermal = np.asarray(thermal.to_value(unit),dtype=np.float64),
            powerlaw = np.asarray(powerlaw.to_value(unit),dtype=np.float64),
            total = np.asarray(total.to_value(unit),dtype=np.float64),
            style=style
        )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "spectrum",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args()

@timed
def main() -> None:
    args = parse_args()
    outpath: Path = args.output
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df_set = FileReader.table_from_csv(args.spectrum)
    dfs = groupby_to_dict(df_set,by=["model_index","t"])
    with PdfPages(outpath) as pdf:
        fig, axes = layout.create()
        model_indices = sorted({k[0] for k in dfs})
        times = sorted({k[1] for k in dfs})

        for i, model_index in enumerate(model_indices):
            handles = []
            for j, time in enumerate(times):
                if (model_index, time) not in dfs:
                    continue
                df_spectrum = dfs[model_index,time]
                print(i,j)
                plotter = pu.Plotter(axes[0,i])
                nu = u.Quantity(
                    df_spectrum["nu"].to_numpy(dtype=np.float64),
                    df_spectrum.attrs["nu_unit"]
                ).to_value(x_axis.label.unit)

                t = u.Quantity(
                    time,
                    df_spectrum.attrs["t_unit"]
                )
                legend = Line2D(
                    [],[],
                    linestyle=styles[j].linestyle,
                    linewidth=styles[j].linewidth,
                    color=styles[j].color,
                    label=f"t={t}",
                )
                handles.append(legend)
                tau = PlotOpticalDepth.from_df(
                    x=nu,
                    df=df_spectrum,
                    style=styles[j]
                )
                lnu = PlotSpecificLuminosity.from_df(
                    x=nu,
                    df=df_spectrum,
                    unit=lnu_axis.label.unit,
                    style=styles[j]
                )
                fnu = PlotSpecificFlux.from_df(
                    x=nu,
                    df=df_spectrum,
                    unit=fnu_axis.label.unit,
                    style=styles[j]
                )
                jnu = PlotEmissivity.from_df(
                    x=nu,
                    df=df_spectrum,
                    unit=jnu_axis.label.unit,
                    style=styles[j]
                )
                anu = PlotAbsorption.from_df(
                    x=nu,
                    df=df_spectrum,
                    unit=anu_axis.label.unit,
                    style=styles[j]
                )
                jnu.draw_thermal(plotter)
                jnu.draw_powerlaw(plotter)
                jnu.draw_total(plotter)
                plotter.ax.legend(
                    handles=handles,
                    fontsize=16
                )
                plotter.title(
                    strings=[
                        f"model {model_index}"
                    ],
                    fontsize=18
                )

                plotter = pu.Plotter(axes[1,i])
                anu.draw_thermal(plotter)
                anu.draw_powerlaw(plotter)
                anu.draw_total(plotter)
                plotter.ax.legend(
                    handles=handles,
                    fontsize=16
                )

                plotter = pu.Plotter(axes[2,i])
                tau.draw_thermal(plotter)
                tau.draw_powerlaw(plotter)
                tau.draw_total(plotter)

                plotter.ax.legend(
                    handles=handles,
                    fontsize=16
                )


                plotter = pu.Plotter(axes[3,i])
                lnu.draw_thermal(plotter)
                lnu.draw_total(plotter)
                plotter.ax.legend(
                    handles=handles,
                    fontsize=16
                )

                plotter = pu.Plotter(axes[4,i])
                fnu.draw_thermal(plotter)
                fnu.draw_total(plotter)
                plotter.ax.legend(
                    handles=handles,
                    fontsize=16
                )

        pdf.savefig(fig)
        plt.close(fig)

if __name__ == "__main__":
    main()
