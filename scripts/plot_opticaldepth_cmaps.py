import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy.typing import NDArray
from module import plot_utils
from module.calculation_lightcurve import ChiSquaredResult
from module.chevalier import BoundaryLabel, ChevalierContours, ObservedRegion, PeakGrid, RelativisticBoundary
from module.models import ModelParameters,DimensionlessThermalLuminosity
from module.mydataclasses import FloatVectorSpace, QuantityArray1D, QuantityData
from module.utils import FileReader, FileWriter
from dataclasses import fields, dataclass
from module import dataframe_accessors
import numpy as np
import astropy.units as u
from module.synchrotron_numerical_table import ThermalSynchrotronIntegrals

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "model_parameter_table",
        "integral_table",
        "plotconfig",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    df_models = FileReader.table_from_csv(args.model_parameter_table)
    model_params = ModelParameters.from_df(df_models)
    plotconfig= plot_utils.PlotConfig.from_yaml(args.plotconfig)

    tau_th_space = FloatVectorSpace(
        start=1.0,
        stop =1.0e+12,
        num = 128,
        scale="log"
    )
    xm_space = FloatVectorSpace(
        start=1.0e-02,
        stop =1.0e+04,
        num = 256,
        scale="log"
    )
    tau_th = tau_th_space.geomspace()
    xm = xm_space.geomspace()
    table = ThermalSynchrotronIntegrals.from_csv(args.integral_table)
    ip = table.ip(xm)

    lambda_th = DimensionlessThermalLuminosity.compute(tau_th,xm,ip)
    peaks = lambda_th.peaks()

    phi_th = u.Quantity(model_params.phi_th,u.dimensionless_unscaled)
    theta0 = model_params.theta0
    logtau = model_params.log10_tauth

    fig, axes = plotconfig.layout.create()
    ax = axes[0, 0]


    mesh = ax.pcolormesh(
        phi_th.to_value("GHz min"),
        theta0,
        logtau,
        shading="auto",
        cmap="viridis",
    )

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r"$\log_{10}\tau_{\rm th}$")

    # Contours
    contours = ax.contour(
        phi_th,
        theta0,
        logtau,
        levels=np.arange(
            np.floor(logtau.min()),
            np.ceil(logtau.max()) + 1,
            2,
        ),
        colors="white",
        linewidths=0.7,
    )

    ax.clabel(contours, inline=True, fontsize=8)

    fig.savefig(outpath)
    plt.close(fig)
if __name__ == "__main__":
    main()
