"""光度曲線を読み込み、power-lawパラメータの比較用にまとめる。

例:
    python plot_powerlaw_lightcurves.py \
        --lightcurves data/lightcurves_powerlaw.csv \
        --model_parameter_table_powerlaw data/model_parameter_table_powerlaw.csv

現在は読み込みとgroupingまで。各groupの概要を表示する。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from module.models import ColumnNames, MetadataNames
from module.mydataclasses import QuantityData
from module.observation import LongformatLightcurve
from module.utils import FileReader
from module import plot_utils
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from module.models import FluxName
from module.mydataclasses import QuantityArray1D

# 比較時に固定する基準値。走査格子にない場合は最も近い値を使う。
REFERENCE_P_INDEX = 3.0
REFERENCE_F_E = 1.0e-2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "plotconfig",
        "powerlaw_lightcurves",
        "model_parameter_table_powerlaw",
        "obslc",
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
        if not isinstance(nu,float):
            raise TypeError(
                f"nuの型が不正です:{type(nu)}"
            )
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


def main() -> None:
    args = parse_args()
    outpath: Path = args.output
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df_lc = FileReader.table_from_csv(args.powerlaw_lightcurves)
    df_powerlaw = FileReader.table_from_csv(args.model_parameter_table_powerlaw)
    df = attach_powerlaw_parameters(df_lc, df_powerlaw)
    lcs_obs = LongformatLightcurve.from_csv(args.obslc)
    frequency_groups = group_by_frequency(df)
    comparisons = (
        (MetadataNames.P_INDEX, MetadataNames.F_E, REFERENCE_F_E),
        (MetadataNames.F_E, MetadataNames.P_INDEX, REFERENCE_P_INDEX),
    )

    plotconf = plot_utils.PlotConfig.from_yaml(args.plotconfig)
    layout = plotconf.layout
    components = plotconf.components
    metadata_config = plotconf.metadata

    # 描画時は2行×len(frequency_groups)列のaxesを用意する。
    with PdfPages(outpath) as pdf:
        fig, axes = layout.create()
        for col, (nu_value, df_frequency) in enumerate(frequency_groups.items()):
            for thermal_model_index, df_thermal in df_frequency.groupby(
                MetadataNames.THERMAL_MODEL_INDEX, sort=True
            ):
                for row, (varied_parameter, fixed_parameter, reference) in enumerate(comparisons):
                    fixed_value, df_group = select_fixed_parameter(
                        df_thermal, fixed_parameter, reference
                    )
                    n_models = df_group[ColumnNames.POWELAW_SUBMODEL_INDEX].nunique()
                    print(
                        f"axes[{row}, {col}]: thermal_model_index={thermal_model_index}, "
                        f"nu={nu_value:g}, {fixed_parameter}={fixed_value:g}, "
                        f"vary {varied_parameter}: {n_models} models"
                    )
                    nu = QuantityData(
                        value=nu_value,
                        unit=df_group.attrs[MetadataNames.NU_UNIT]
                    )

                    axis_config = plotconf.layout.axes.for_subplot(row,col)
                    plotter = plot_utils.Plotter(axes[row,col])
                    x_unit:str = axis_config.x_unit
                    y_unit:str = axis_config.y_unit
                    lc_obs = lcs_obs.select_frequency(nu, unit=nu.unit)
                    fnu_background = lc_obs.fnu_background.to_value_in(y_unit)

                    # observed lightcurve
                    t_obs = lc_obs.t.to_value_in(x_unit)
                    plotter.line(
                        x=t_obs,
                        y=fnu_background,
                        style=components["background"].resolve_line_style(col)
                    )
                    plotter.errorbar(
                        x=t_obs,
                        y=lc_obs.fnu_total.to_value_in(y_unit),
                        yerr=lc_obs.fnu_total_err.to_value_in(y_unit),
                        style=components["observed"].resolve_errorbar_style(col)
                    )
                    dfs = df_group.groupby(ColumnNames.POWELAW_SUBMODEL_INDEX)

                    for i, (submodel_index, df_model) in enumerate(
                        df_group.groupby(ColumnNames.POWELAW_SUBMODEL_INDEX, sort=True)
                    ):
                        df_model = df_model.sort_values(ColumnNames.T).copy()
                        df_model.attrs = df_group.attrs.copy()

                        parameter_value = float(df_model[varied_parameter].iloc[0])

                        # このdf_modelが、axes[row, col]に描く1本の光度曲線
                        t = QuantityArray1D.from_df(
                            df_model,
                            ColumnNames.T,
                            unit_key=MetadataNames.T_UNIT,
                        )
                        fnu = QuantityArray1D.from_df(
                            df_model,
                            FluxName.FNU_TOTAL,
                            unit_key=MetadataNames.FNU_UNIT,
                        )

                        background = lc_obs.fnu_background.index(0).value_in(y_unit)

                        plotter.line(
                            x=t.to_value_in(x_unit),
                            y=fnu.to_value_in(y_unit) + background,
                            style=components["continuous"].resolve_line_style(col),
                        )
        pdf.savefig(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
