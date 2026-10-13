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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from module import plot_utils
from module.models import ColumnNames, ModelParameters
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


def plot_ratios(df: pd.DataFrame, plotconf: plot_utils.PlotConfig, output: Path) -> None:
    """Save a single-panel ratio plot with a threshold at one."""
    indices = df[ColumnNames.MODEL_INDEX].to_numpy(dtype=np.int64)
    ratio = df["peak_cutoff_ratio"].to_numpy(dtype=np.float64)
    failed_count = int(np.count_nonzero(ratio <= 1.0))
    left, right = float(indices.min()), float(indices.max())
    if left == right:
        left, right = left - 0.5, right + 0.5
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        fig, axes = plotconf.layout.create()
        try:
            if axes.shape != (1, 1):
                raise ValueError("plotconfigのlayoutは1行1列にしてください")
            plotter = plot_utils.Plotter(axes[0, 0])
            components = plotconf.components
            ratio_line = plotter.line(
                x=indices,
                y=ratio,
                style=components["ratio"].resolve_line_style(0),
            )
            threshold_line = plotter.line(
                x=np.array([left, right]),
                y=np.ones(2),
                style=components["threshold"].resolve_line_style(0),
            )
            plotter.legend(
                entries=[
                    plot_utils.LegendEntry(components["ratio"], ratio_line[0]),
                    plot_utils.LegendEntry(components["threshold"], threshold_line[0]),
                ],
                config=components["legend"],
            )
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
    plot_ratios(df, plotconf, args.output)
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
