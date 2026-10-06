"""光度曲線を読み込み、power-lawパラメータの比較用にまとめる。

例:
    python plot_powerlaw_lightcurves.py \
        --powerlaw_lightcurves data/lightcurves_powerlaw.csv \
        --model_parameter_table_powerlaw data/model_parameter_table_powerlaw.csv

1行目はp_index、2行目はf_eを色で区別して描画する。
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LogNorm, to_hex

from module import plot_utils
from module.calculation_lightcurve import BinnedLightcurveWithBackground
from module.models import ColumnNames, FluxName, MetadataNames
from module.mydataclasses import QuantityArray1D, QuantityData,LongformatDataFrame
from module.observation import LongformatLightcurve
from module.utils import FileReader

# 比較時に固定する基準値。走査格子にない場合は最も近い値を使う。
REFERENCE_P_INDEX = 2.5
REFERENCE_F_E = 1.0e-1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "plotconfig", "powerlaw_lightcurves", "model_parameter_table_powerlaw",
        "obslc", "output",
        "candidate_binned_lightcurves",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args()


def attach_powerlaw_parameters(
    df_lc: pd.DataFrame,
    df_powerlaw: pd.DataFrame,
) -> pd.DataFrame:
    """各光度曲線の行にp_indexとf_eを結合し、単位情報を引き継ぐ。"""
    keys = [
        ColumnNames.POWELAW_SUBMODEL_INDEX,
        MetadataNames.THERMAL_MODEL_INDEX,
        ColumnNames.NU,
    ]
    parameters = [MetadataNames.P_INDEX, MetadataNames.F_E]
    df = df_lc.merge(
        df_powerlaw.loc[:, keys + parameters],
        on=keys,
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if (df["_merge"] != "both").any():
        raise ValueError("光度曲線に対応するpower-lawパラメータがありません")
    df = df.drop(columns="_merge")
    df.attrs = df_lc.attrs.copy()
    return df


def group_by_frequency(df: pd.DataFrame) -> dict[float, pd.DataFrame]:
    """周波数をキーとする辞書を作る。各DataFrameに単位情報を残す。"""
    groups: dict[float, pd.DataFrame] = {}
    for nu, df_frequency in df.groupby(ColumnNames.NU, sort=True):
        df_frequency.attrs = df.attrs.copy()
        if not isinstance(nu, float):
            raise TypeError(f"nuの型が不正です: {type(nu)}")
        groups[float(nu)] = df_frequency
    return groups


def select_fixed_parameter(
    df: pd.DataFrame,
    parameter: str,
    reference_value: float,
) -> tuple[float, pd.DataFrame]:
    """基準値に最も近い格子値を固定した比較データを返す。"""
    values = df[parameter].dropna().to_numpy(dtype=float)
    if len(values) == 0:
        raise ValueError(f"{parameter}の格子値がありません")
    fixed_value = float(values[abs(values - reference_value).argmin()])
    selected = df.loc[df[parameter] == fixed_value].copy()
    selected.attrs = df.attrs.copy()
    return fixed_value, selected


def require_numeric_scalar(value: object) -> float:
    """Python・NumPyの数値スカラーであることを確認してfloatに変換する。"""
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"数値スカラーが必要です: {type(value)}")
    return float(value)


def parameter_color(mappable: ScalarMappable, value: float) -> str:
    """1要素の配列から色を求め、型が確定したRGBAタプルへ変換する。"""
    colors = mappable.to_rgba(np.array([value], dtype=np.float64))
    rgba: tuple[float, float, float, float] = (
        require_numeric_scalar(colors[0, 0]),
        require_numeric_scalar(colors[0, 1]),
        require_numeric_scalar(colors[0, 2]),
        require_numeric_scalar(colors[0, 3]),
    )
    return to_hex(rgba)


def build_color_maps(df: pd.DataFrame) -> dict[str, ScalarMappable]:
    """全周波数で共通の、パラメータ値と色の対応を作る。"""
    if df.empty:
        raise ValueError("描画する光度曲線がありません")
    if (df[MetadataNames.F_E] <= 0).any():
        raise ValueError("f_eの対数カラースケールには正の値が必要です")
    return {
        MetadataNames.P_INDEX: ScalarMappable(
            norm=Normalize(
                vmin=require_numeric_scalar(df[MetadataNames.P_INDEX].min()),
                vmax=require_numeric_scalar(df[MetadataNames.P_INDEX].max()),
            ),
            cmap="viridis",
        ),
        MetadataNames.F_E: ScalarMappable(
            norm=LogNorm(
                vmin=require_numeric_scalar(df[MetadataNames.F_E].min()),
                vmax=require_numeric_scalar(df[MetadataNames.F_E].max()),
            ),
            cmap="viridis",
        ),
    }


def main() -> None:
    args = parse_args()
    outpath: Path = args.output
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df_lc = FileReader.table_from_csv(args.powerlaw_lightcurves)
    df_powerlaw = FileReader.table_from_csv(args.model_parameter_table_powerlaw)
    df = attach_powerlaw_parameters(df_lc, df_powerlaw)
    frequency_groups = group_by_frequency(df)
    color_maps = build_color_maps(df)
    lcs_obs = LongformatLightcurve.from_csv(args.obslc)
    longdf_binned = LongformatDataFrame.read_from(args.candidate_binned_lightcurves)
    dfs_binned_nu = longdf_binned.parse(ColumnNames.NU)
    comparisons = (
        (MetadataNames.P_INDEX, MetadataNames.F_E, REFERENCE_F_E),
        (MetadataNames.F_E, MetadataNames.P_INDEX, REFERENCE_P_INDEX),
    )
    plotconf = plot_utils.PlotConfig.from_yaml(args.plotconfig)
    components = plotconf.components


    with PdfPages(outpath) as pdf:
        fig, axes = plotconf.layout.create()
        try:
            if axes.shape != (2, len(frequency_groups)):
                raise ValueError(
                    f"plotconfigのaxesは2行×{len(frequency_groups)}列にしてください: "
                    f"現在のshape={axes.shape}"
                )
            for col, (nu_value, df_frequency) in enumerate(frequency_groups.items()):
                nu = QuantityData(
                    value=nu_value,
                    unit=df_frequency.attrs[MetadataNames.NU_UNIT],
                )
                lc_obs = lcs_obs.select_frequency(nu, unit=nu.unit)
                for row, (varied_parameter, fixed_parameter, reference) in enumerate(comparisons):
                    axis_config = plotconf.layout.axes.for_subplot(row, col)
                    plotter = plot_utils.Plotter(axes[row, col])
                    x_unit: str = axis_config.x_unit
                    y_unit: str = axis_config.y_unit
                    background = lc_obs.fnu_background.index(0).value_in(y_unit)
                    mappable = color_maps[varied_parameter]
                    base_style = components["continuous"].resolve_line_style(0)

                    plotter.line(
                        x=lc_obs.t.to_value_in(x_unit),
                        y=lc_obs.fnu_background.to_value_in(y_unit),
                        style=components["background"].resolve_line_style(col),
                    )
                    plotter.errorbar(
                        x=lc_obs.t.to_value_in(x_unit),
                        y=lc_obs.fnu_total.to_value_in(y_unit),
                        yerr=lc_obs.fnu_total_err.to_value_in(y_unit),
                        style=components["observed"].resolve_errorbar_style(col),
                    )

                    for thermal_model_index, df_thermal in df_frequency.groupby(
                        MetadataNames.THERMAL_MODEL_INDEX, sort=True
                    ):
                        fixed_value, df_group = select_fixed_parameter(
                            df_thermal, fixed_parameter, reference
                        )
                        df_binned_nu = dfs_binned_nu[nu_value]
                        df_binned_plot: pd.DataFrame = df_binned_nu.loc[
                            df_binned_nu[ColumnNames.MODEL_INDEX] == thermal_model_index
                        ]
                        df_binned_plot.attrs[ColumnNames.NU] = nu_value
                        df_binned_plot.attrs[ColumnNames.MODEL_INDEX] = thermal_model_index
                        binned_lc = BinnedLightcurveWithBackground.from_df(df_binned_plot)

                        for submodel_index, df_model in df_group.groupby(
                            ColumnNames.POWELAW_SUBMODEL_INDEX, sort=True
                        ):
                            df_model = df_model.sort_values(ColumnNames.T).copy()
                            df_model.attrs = df_group.attrs.copy()
                            parameter_value = float(df_model[varied_parameter].iloc[0])
                            style = replace(
                                base_style,
                                color=parameter_color(mappable, parameter_value),
                            )
                            t = QuantityArray1D.from_df(
                                df_model, ColumnNames.T,
                                unit_key=MetadataNames.T_UNIT,
                            )
                            fnu = QuantityArray1D.from_df(
                                df_model, FluxName.FNU_TOTAL,
                                unit_key=MetadataNames.FNU_UNIT,
                            )
                            plotter.line(
                                x=t.to_value_in(x_unit),
                                y=fnu.to_value_in(y_unit) + background,
                                style=style,
                            )
                            # binned lightcurve
                            plotter.step(
                                edges=binned_lc.bin_edges.to_value_in(x_unit),
                                values=binned_lc.fnu_with_bg.to_value_in(y_unit),
                                style=components["binned"].resolve_line_style(col),
                                baseline=background
                            )
                        #fiducial_parameter
                        plotter.text(
                            x=0.98,y=0.98,
                            label=f"{fixed_parameter}={fixed_value:.1f}",
                            style=components["fiducial_parameter"].text_style_config,
                            coordinates="axes"
                        )
            colorbar_names = ("p_index_colorbar", "f_e_colorbar")
            for row, (varied_parameter, _, _) in enumerate(comparisons):
                colorbar = plot_utils.ColorbarComponent(
                    mappable=color_maps[varied_parameter],
                    config=components[colorbar_names[row]].colorbar_style_config,
                )
                colorbar.draw(
                    fig=fig,
                    axes=axes[row, :].tolist(),
                )
            pdf.savefig(fig)
        finally:
            plt.close(fig)


if __name__ == "__main__":
    main()
