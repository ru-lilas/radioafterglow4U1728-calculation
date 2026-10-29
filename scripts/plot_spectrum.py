import argparse
from pathlib import Path
from typing import Any, Hashable, Literal, TypeGuard, get_args
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
jnu_ratio_axis = replace(
    x_axis,
    limits=(1.0e-04,1.0e+04),
    label = pu.LabelConfigure(
        prefix=r'$j_{\nu,\mathrm{pl}}~/~j_{\nu,\mathrm{th}}$',
        unit="",
        fontsize=32
    )
)
anu_ratio_axis = replace(
    jnu_ratio_axis,
    label = pu.LabelConfigure(
        prefix=r'$\alpha_{\nu,\mathrm{pl}}~/~\alpha_{\nu,\mathrm{th}}$',
        unit="",
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
        fnu_axis,
        jnu_ratio_axis,
        anu_ratio_axis
    ),
    ticks=ticks
)
NROWS=len(AXES.y)
NCOLS=2
figure = pu.FigureConfig(
    figsize=(12*NCOLS,9*NROWS)
)
subplots = pu.SubplotConfig(
    nrows=NROWS,
    ncols=NCOLS,
    sharex="col",
    sharey="row",
    hide_shared_labels=True
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

    @classmethod
    def from_df(
        cls,
        x:NDArray[np.float64],
        df:pd.DataFrame,
        quantity_name:str,
        unit: str,
        style: pu.LineStyle
    ):
        df_unit = df.attrs[f"{quantity_name}_unit"]
        thermal = u.Quantity(
            df[f"{quantity_name}_thermal"].to_numpy(dtype=np.float64),
            df_unit
        )
        powerlaw = u.Quantity(
            df[f"{quantity_name}_powerlaw"].to_numpy(dtype=np.float64),
            df_unit
        )
        total = u.Quantity(
            df[f"{quantity_name}_total"].to_numpy(dtype=np.float64),
            df_unit
        )
        return cls(
            x=x,
            thermal = np.asarray(thermal.to_value(unit),dtype=np.float64),
            powerlaw = np.asarray(powerlaw.to_value(unit),dtype=np.float64),
            total = np.asarray(total.to_value(unit),dtype=np.float64),
            style=style
        )

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
    def from_df_dimless(
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
class PlotEmissivity(PlotComponentBase):
    @property
    def ratio(
        self
    )->NDArray[np.float64]:
        return self.powerlaw/self.thermal

    def draw_ratio(
        self,
        plotter:pu.Plotter,
        style:pu.LineStyle
    ):
        plotter.ax.axhline(
            y=1.0e-02,
            lw=0.5,
            ls="--",
            color="#000000"
        )
        plotter.ax.axhline(
            y=1.0e+02,
            lw=0.5,
            ls="--",
            color="#000000"
        )
        return plotter.line(
            self.x,
            self.ratio,
            style=style
        )
        

@dataclass(frozen=True,slots=True)
class PlotAbsorption(PlotComponentBase):

    @property
    def ratio(
        self
    )->NDArray[np.float64]:
        return self.powerlaw/self.thermal

    def draw_ratio(
        self,
        plotter:pu.Plotter,
        style:pu.LineStyle
    ):
        plotter.ax.axhline(
            y=1.0e-02,
            lw=0.5,
            ls="--",
            color="#000000"
        )
        plotter.ax.axhline(
            y=1.0e+02,
            lw=0.5,
            ls="--",
            color="#000000"
        )
        return plotter.line(
            self.x,
            self.ratio,
            style=style
        )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "spectrum",
        "model_parameter_table",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args()

def is_float64_array(value: object) -> TypeGuard[NDArray[np.float64]]:
    return (
        isinstance(value, np.ndarray)
        and value.dtype == np.dtype("float64")
    )

@timed
def main() -> None:
    args = parse_args()
    outpath: Path = args.output
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df_set = FileReader.table_from_csv(args.spectrum)
    df_models = FileReader.table_from_csv(args.model_parameter_table)
    models = ModelParameters.from_df(df_models)
    dfs = groupby_to_dict(df_set,by=["model_index","t"])
    with PdfPages(outpath) as pdf:
        fig, axes = layout.create()
        model_indices = sorted({k[0] for k in dfs})
        times = sorted({k[1] for k in dfs})

        for i, model_index in enumerate(model_indices):
            handles = []
            model = models.select_models([model_index])
            f_e = model.powerlaw.f_e
            for j, time in enumerate(times):
                if (model_index, time) not in dfs:
                    continue
                df_spectrum = dfs[model_index,time]
                plotter = pu.Plotter(axes[0,i])
                nu = u.Quantity(
                    df_spectrum["nu"].to_numpy(dtype=np.float64),
                    df_spectrum.attrs["nu_unit"]
                ).to_value(x_axis.label.unit)
                if not is_float64_array(nu):
                    raise TypeError(
                        "nuの型が不正です"
                    )

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
                tau = PlotOpticalDepth.from_df_dimless(
                    x=nu,
                    df=df_spectrum,
                    style=styles[j]
                )
                lnu = PlotComponentBase.from_df(
                    x=nu,
                    df=df_spectrum,
                    unit=lnu_axis.label.unit,
                    quantity_name="lnu",
                    style=styles[j]
                )
                fnu = PlotComponentBase.from_df(
                    x=nu,
                    df=df_spectrum,
                    quantity_name="fnu",
                    unit=fnu_axis.label.unit,
                    style=styles[j]
                )
                jnu = PlotEmissivity.from_df(
                    x=nu,
                    df=df_spectrum,
                    quantity_name="jnu",
                    unit=jnu_axis.label.unit,
                    style=styles[j]
                )
                anu = PlotAbsorption.from_df(
                    x=nu,
                    df=df_spectrum,
                    quantity_name="alpha_nu",
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
                plotter.ax.axvline(
                    9e+9,
                    color="#000000",
                    linestyle="--",
                    linewidth=0.5
                )
                plotter.title(
                    strings=[
                        f"model {model_index}",
                        rf'$\varepsilon_B$={model.eps_b:.1e}',
                        rf'$\varepsilon_\mathrm{{th}}$={model.eps_th:.1f}',
                        rf'$f_e$={f_e:.1e}',
                        rf'$p$={model.powerlaw.p_index:.2f}'
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
                plotter.ax.axvline(
                    9e+9,
                    color="#000000",
                    linestyle="--",
                    linewidth=0.5
                )

                plotter = pu.Plotter(axes[2,i])
                tau.draw_thermal(plotter)
                tau.draw_powerlaw(plotter)
                tau.draw_total(plotter)
                plotter.ax.axhline(1.0,color="#000000")

                plotter.ax.legend(
                    handles=handles,
                    fontsize=16
                )
                plotter.ax.axvline(
                    9e+9,
                    color="#000000",
                    linestyle="--",
                    linewidth=0.5
                )


                plotter = pu.Plotter(axes[3,i])
                lnu.draw_thermal(plotter)
                lnu.draw_total(plotter)
                plotter.ax.legend(
                    handles=handles,
                    fontsize=16
                )
                plotter.ax.axvline(
                    9e+9,
                    color="#000000",
                    linestyle="--",
                    linewidth=0.5
                )

                plotter = pu.Plotter(axes[4,i])
                fnu.draw_thermal(plotter)
                fnu.draw_total(plotter)
                plotter.ax.legend(
                    handles=handles,
                    fontsize=16
                )
                plotter.ax.axvline(
                    9e+9,
                    color="#000000",
                    linestyle="--",
                    linewidth=0.5
                )

                plotter = pu.Plotter(axes[5,i])
                jnu.draw_ratio(
                    plotter,
                    style=styles[j]
                )
                plotter.ax.axvline(
                    9e+9,
                    color="#000000",
                    linestyle="--",
                    linewidth=0.5
                )

                plotter = pu.Plotter(axes[6,i])
                anu.draw_ratio(
                    plotter,
                    style=styles[j]
                )
                plotter.ax.axvline(
                    9e+9,
                    color="#000000",
                    linestyle="--",
                    linewidth=0.5
                )
        pdf.savefig(fig)
        plt.close(fig)

if __name__ == "__main__":
    main()
