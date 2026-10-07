"""x_Mに対するzeta・etaのthermal、powerlaw、total成分を描画する。

例:
    python plot_dimensionless_synchrotrons.py \
        --plotconfig dimensionless_synchrotrons.yaml \
        --powerlaw_lightcurves data/lightcurves_powerlaw.csv \
        --model_parameter_table_powerlaw data/model_parameter_table_powerlaw.csv \
        --output reports/dimensionless_synchrotrons.pdf

各熱モデルにつきthermal、powerlaw、totalの3列を使い、2モデルで6列にする。
行はzeta、eta、zeta/eta、無次元光度の順。
p_indexとf_eの比較を別ページへ保存する。
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
from module.models import ColumnNames, DimensionlessSynchrotronKeys, MetadataNames
from module.utils import FileReader

# 比較時に固定する基準値。走査格子にない場合は最も近い値を使う。
REFERENCE_P_INDEX = 2.5
REFERENCE_F_E = 1.0e-1

# 上からzeta、eta、source functionの周波数依存項を描く。
QUANTITY_ORDER = ("zeta", "eta", "zeta/eta", "lambda")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "plotconfig", "powerlaw_lightcurves", "model_parameter_table_powerlaw",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args()


def attach_powerlaw_parameters(
    df_lc: pd.DataFrame,
    df_powerlaw: pd.DataFrame,
) -> pd.DataFrame:
    """各光度曲線の行にp_indexとf_eを結合し、単位情報を引き継ぐ。"""
    keys = [
        ColumnNames.POWERLAW_SUBMODEL_INDEX,
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


def group_by_thermal_model(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """thermal_model_indexの昇順でモデルごとの辞書を作る。"""
    groups: dict[int, pd.DataFrame] = {}
    for model_index, df_model in df.groupby(MetadataNames.THERMAL_MODEL_INDEX, sort=True):
        value = require_numeric_scalar(model_index)
        if not np.isfinite(value) or not value.is_integer():
            raise ValueError(f"thermal_model_indexには整数値が必要です: {value}")
        df_model.attrs = df.attrs.copy()
        groups[int(value)] = df_model
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
    """全モデルで共通の、パラメータ値と色の対応を作る。"""
    if df.empty:
        raise ValueError("描画する無次元量がありません")
    if (df[MetadataNames.F_E] <= 0).any():
        raise ValueError("f_eの対数カラースケールには正の値が必要です")
    return {
        MetadataNames.P_INDEX: ScalarMappable(
            norm=Normalize(
                vmin=require_numeric_scalar(df[MetadataNames.P_INDEX].min()),
                vmax=require_numeric_scalar(df[MetadataNames.P_INDEX].max()),
            ),
            cmap="cividis",
        ),
        MetadataNames.F_E: ScalarMappable(
            norm=LogNorm(
                vmin=require_numeric_scalar(df[MetadataNames.F_E].min()),
                vmax=require_numeric_scalar(df[MetadataNames.F_E].max()),
            ),
            cmap="cividis",
        ),
    }


def main() -> None:
    args = parse_args()
    outpath: Path = args.output
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df_syn = FileReader.table_from_csv(args.powerlaw_lightcurves)
    df_powerlaw = FileReader.table_from_csv(args.model_parameter_table_powerlaw)
    df = attach_powerlaw_parameters(df_syn, df_powerlaw)
    thermal_groups = group_by_thermal_model(df)
    if len(thermal_groups) != 2:
        raise ValueError(
            f"6列の描画には2つのthermal_model_indexが必要です: {list(thermal_groups)}"
        )
    color_maps = build_color_maps(df)

    keys = DimensionlessSynchrotronKeys
    quantity_columns = {
        "zeta": (keys.ZETA_THERMAL, keys.ZETA_POWERLAW, keys.ZETA_TOTAL),
        "eta": (keys.ETA_THERMAL, keys.ETA_POWERLAW, keys.ETA_TOTAL),
        "zeta/eta": (keys.SNU_THERMAL, keys.SNU_POWERLAW, keys.SNU_TOTAL),
        "lambda": (keys.LAMBDA_THERMAL, keys.LAMBDA_POWERLAW, keys.LAMBDA_TOTAL),
    }
    required_columns = [column for columns in quantity_columns.values() for column in columns]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"無次元量の列がありません: {missing}。更新したmodels.pyで入力CSVを再生成してください"
        )
    comparisons = (
        (MetadataNames.P_INDEX, MetadataNames.F_E, REFERENCE_F_E),
        (MetadataNames.F_E, MetadataNames.P_INDEX, REFERENCE_P_INDEX),
    )
    plotconf = plot_utils.PlotConfig.from_yaml(args.plotconfig)
    components = plotconf.components

    with PdfPages(outpath) as pdf:
        for varied_parameter, fixed_parameter, reference in comparisons:
            fig, axes = plotconf.layout.create()
            try:
                if axes.shape != (len(QUANTITY_ORDER), 6):
                    raise ValueError(
                        "plotconfigのaxesは4行×6列にしてください: "
                        f"現在のshape={axes.shape}"
                    )
                fig.suptitle(f"Varying {varied_parameter}")
                mappable = color_maps[varied_parameter]
                component_names = ("thermal", "powerlaw", "total")
                for model_col, (thermal_model_index, df_thermal) in enumerate(thermal_groups.items()):
                    fixed_value, df_group = select_fixed_parameter(
                        df_thermal, fixed_parameter, reference
                    )
                    for component_col, component_name in enumerate(component_names):
                        col = 3*model_col + component_col
                        axes[0, col].set_title(
                            f"thermal_model_index={thermal_model_index}\n"
                            f"{component_name}, {fixed_parameter}={fixed_value:.3g}"
                        )
                        base_style = components[component_name].resolve_line_style(0)
                        for row, quantity_name in enumerate(QUANTITY_ORDER):
                            column = quantity_columns[quantity_name][component_col]
                            plotter = plot_utils.Plotter(axes[row, col])
                            for i, (submodel_index, df_model) in enumerate(
                                df_group.groupby(ColumnNames.POWERLAW_SUBMODEL_INDEX, sort=True)
                            ):
                                # thermal成分は走査パラメータによらないので1本だけ描く。
                                if component_name == "thermal" and i > 0:
                                    continue
                                df_model = df_model.sort_values(ColumnNames.XM)
                                xm = df_model[ColumnNames.XM].to_numpy(dtype=np.float64)
                                style = base_style
                                if component_name != "thermal":
                                    parameter_value = require_numeric_scalar(
                                        df_model[varied_parameter].iloc[0]
                                    )
                                    style = replace(
                                        base_style,
                                        color=parameter_color(mappable, parameter_value),
                                    )
                                plotter.line(
                                    x=xm,
                                    y=df_model[column].to_numpy(dtype=np.float64),
                                    style=style,
                                )

                colorbar_name = (
                    "p_index_colorbar"
                    if varied_parameter == MetadataNames.P_INDEX
                    else "f_e_colorbar"
                )
                for row in range(len(QUANTITY_ORDER)):
                    colorbar = plot_utils.ColorbarComponent(
                        mappable=mappable,
                        config=components[colorbar_name].colorbar_style_config,
                    )
                    colorbar.draw(fig=fig, axes=axes[row, :].tolist())
                pdf.savefig(fig)
            finally:
                plt.close(fig)


if __name__ == "__main__":
    main()
