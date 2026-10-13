"""Plot and verify xm_peak/xm_cutoff for every model.

Example:
    python plot_peak_cutoff_ratio.py \
        --peak_grid data/peak_grid.csv \
        --model_parameter_table data/model_parameter_table.csv \
        --plotconfig peak_cutoff_ratio.yaml \
        --output reports/peak_cutoff_ratio.pdf
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from numpy.typing import NDArray

from module import plot_utils
from module.models import ColumnNames, ModelParameters
from module.mydataclasses import QuantityData
from module.utils import FileReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("peak_grid", "model_parameter_table", "plotconfig", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args()


def build_ratio_table(
    df_peak: pd.DataFrame,
    model_params: ModelParameters,
) -> pd.DataFrame:
    """Match every model by ID and calculate its peak-to-cutoff ratio."""
    df_cutoff = pd.DataFrame({
        ColumnNames.MODEL_INDEX: model_params.model_indices.ravel(),
        "xm_cutoff": model_params.xm_cutoff.ravel(),
        ColumnNames.PHI_TH: model_params.phi_th.ravel(),
        ColumnNames.THETA0: model_params.theta0.ravel(),
    })
    df = df_peak.loc[:, [ColumnNames.MODEL_INDEX, ColumnNames.XM]].rename(
        columns={ColumnNames.XM: "xm_peak"},
    )
    if df[ColumnNames.MODEL_INDEX].isna().any():
        raise ValueError("ピーク表のmodel_indexに欠損があります")
    df = df.merge(
        df_cutoff,
        on=ColumnNames.MODEL_INDEX,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if df.empty:
        raise ValueError("モデルがありません")
    if (df["_merge"] != "both").any():
        raise ValueError("ピーク表とモデル表のmodel_indexが一致しません")
    df = df.drop(columns="_merge")
    peak = df["xm_peak"].to_numpy(dtype=np.float64)
    cutoff = df["xm_cutoff"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(peak) & (peak > 0)):
        raise ValueError("xm_peakは正の有限値が必要です")
    if not np.all(np.isfinite(cutoff) & (cutoff > 0)):
        raise ValueError("xm_cutoffは正の有限値が必要です")
    ratio = peak / cutoff
    if not np.all(np.isfinite(ratio) & (ratio > 0)):
        raise ValueError("xm_peak/xm_cutoffは正の有限値が必要です")
    df["peak_cutoff_ratio"] = ratio
    return df.sort_values(ColumnNames.MODEL_INDEX).reset_index(drop=True)


def report_ratios(df: pd.DataFrame) -> None:
    """Report a strict ratio > 1 check, including all failing models."""
    ratio = df["peak_cutoff_ratio"].to_numpy(dtype=np.float64)
    passed = ratio > 1.0
    print(f"Models with ratio > 1: {np.count_nonzero(passed)}/{len(df)}")
    print(f"Minimum ratio: {ratio.min():.6g}")
    if np.all(passed):
        print("PASS: every model has xm_peak/xm_cutoff > 1.")
    else:
        print("FAIL: models with xm_peak/xm_cutoff <= 1:")
        print(df.loc[~passed].to_string(index=False))


def build_ratio_grid(
    df: pd.DataFrame,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return phi, theta0 and ratios with rows for theta0 and columns for phi."""
    grid = df.pivot(
        index=ColumnNames.THETA0,
        columns=ColumnNames.PHI_TH,
        values="peak_cutoff_ratio",
    ).sort_index().sort_index(axis=1)
    if grid.isna().to_numpy().any():
        raise ValueError("phi_thとtheta0のモデル格子に欠落があります")
    return (
        grid.columns.to_numpy(dtype=np.float64),
        grid.index.to_numpy(dtype=np.float64),
        grid.to_numpy(dtype=np.float64),
    )


def model_title_strings(
    model_params: ModelParameters,
    plotconf: plot_utils.PlotConfig,
) -> list[str]:
    """Format shared model parameters for the figure title."""
    quantities = {
        "eps_b": QuantityData(model_params.eps_b, ""),
        "eps_th": QuantityData(model_params.eps_th, ""),
        "mu_e_mu": QuantityData(model_params.molecular_weight.to_ratio(), ""),
        "f_e": QuantityData(model_params.powerlaw.f_e, ""),
        "p_index": QuantityData(model_params.powerlaw.p_index, ""),
        "distance": model_params.distance,
    }
    strings: list[str] = []
    for name, quantity in quantities.items():
        plot_utils.MetadataComponent(
            quantity=quantity,
            config=plotconf.metadata.items[name],
        ).add_strings(strings)
    return strings


def plot_ratios(
    df: pd.DataFrame,
    plotconf: plot_utils.PlotConfig,
    output: Path,
    title_strings: list[str],
) -> None:
    """Plot log10 of the ratio with ratio=1 at the diverging color center."""
    ratio = df["peak_cutoff_ratio"].to_numpy(dtype=np.float64)
    failed_count = int(np.count_nonzero(ratio <= 1.0))
    phi_axis, theta0_axis, ratio_grid = build_ratio_grid(df)
    log_ratio = np.log10(ratio_grid)
    style = plotconf.components["ratio_map"].colormap_style_config
    if style.norm.type != "centered" or style.norm.vcenter != 0.0:
        raise ValueError("ratio_mapはnorm.type=centered、vcenter=0にしてください")
    # Preserve both colors even when all ratios lie on one side of one.
    limit = max(float(np.max(np.abs(log_ratio))), 1.0)
    style = replace(
        style,
        vmin=-limit if style.vmin is None else style.vmin,
        vmax=limit if style.vmax is None else style.vmax,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        fig, axes = plotconf.layout.create()
        try:
            if axes.shape != (1, 1):
                raise ValueError("plotconfigのlayoutは1行1列にしてください")
            fig.suptitle(
                ", ".join(title_strings),
                fontsize=plotconf.metadata.fontsize,
            )
            axis_config = plotconf.layout.axes.for_subplot(0, 0)
            phi_values = np.asarray(
                u.Quantity(phi_axis, unit=u.dimensionless_unscaled).to_value(
                    axis_config.x_unit or u.dimensionless_unscaled
                ),
                dtype=np.float64,
            )
            plotter = plot_utils.Plotter(axes[0, 0])
            mesh = plotter.colormap(
                x=phi_values,
                y=theta0_axis,
                z=log_ratio,
                style=style,
            )
            colorbar = plot_utils.ColorbarComponent(mappable=mesh, config=style.colorbar)
            colorbar.draw(fig=fig, axes=[axes[0, 0]])
            plotter.title(
                strings=[
                    f"Minimum ratio: {ratio.min():.3g}",
                    f"Models with ratio <= 1: {failed_count}/{len(df)}",
                ],
                fontsize=plotconf.metadata.fontsize,
            )
            pdf.savefig(fig)
        finally:
            plt.close(fig)


def main() -> None:
    args = parse_args()
    print("Plotting peak-to-cutoff ratios...", flush=True)
    df_peak = FileReader.table_from_csv(args.peak_grid)
    df_model = FileReader.table_from_csv(args.model_parameter_table)
    model_params = ModelParameters.from_df(df_model)
    df = build_ratio_table(df_peak, model_params)
    report_ratios(df)
    plotconf = plot_utils.PlotConfig.from_yaml(args.plotconfig)
    plot_ratios(
        df, plotconf, args.output,
        model_title_strings(model_params, plotconf),
    )
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
