"""thermal-onlyの推定結果にp_index × f_eを組み合わせる。

実行例（プロジェクトのscriptsディレクトリへ配置した場合）:
    python -m scripts.model_parameter_table_powerlaw \
        --input data/run_000001/input.yaml \
        --minimum_chi_squared data/run_000001/minimum_chi_squared.csv \
        --output data/run_000001/model_parameter_table_powerlaw.csv

input.yamlからpowerlaw_sensitivityの探索範囲を取得する。
thermalモデルのパラメータ表との照合は、光度曲線の計算時に行う。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from module.inputs_as_dataclass import GeneralInputs
from module.models import ColumnNames
from module.utils import FileReader, FileWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "input",
        "minimum_chi_squared",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args()


def build_parameter_table(
    inputs: GeneralInputs,
    df_minimum: pd.DataFrame,
) -> pd.DataFrame:
    if df_minimum.empty:
        raise ValueError("minimum_chi_squaredにthermalモデルがありません")

    minimum = df_minimum.loc[:, ["model_index", "chi_squared", "ndof", "nu"]].copy()
    numeric_indices = pd.to_numeric(minimum["model_index"], errors="raise")
    if not isinstance(numeric_indices, pd.Series):
        raise TypeError("model_indexの数値変換結果はpd.Seriesである必要があります")
    indices = numeric_indices.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(indices) & (indices == np.floor(indices))):
        raise ValueError("thermal model_indexには整数値が必要です")
    minimum["model_index"] = indices.astype(np.int64)
    minimum = minimum.rename(columns={
        ColumnNames.MODEL_INDEX: ColumnNames.THERMAL_MODEL_INDEX,
        "chi_squared": "thermal_chi_squared",
        "ndof": "thermal_ndof",
    })

    sensitivity = inputs.powerlaw_sensitivity
    p_indices = sensitivity.p_indices.to_floatvector().values
    f_es = sensitivity.f_es.to_floatvector().values
    p_grid, f_grid = np.meshgrid(p_indices, f_es, indexing="ij")
    powerlaw = pd.DataFrame({
        "p_index": p_grid.ravel(),
        "f_e": f_grid.ravel(),
    })
    if powerlaw.empty:
        raise ValueError("power-lawの探索格子が空です")

    df = minimum.merge(powerlaw, how="cross")
    df.insert(0, ColumnNames.POWERLAW_SUBMODEL_INDEX, np.arange(len(df), dtype=np.int64))

    df = df.loc[:, [
        ColumnNames.POWERLAW_SUBMODEL_INDEX,
        ColumnNames.THERMAL_MODEL_INDEX,
        "nu", "p_index", "f_e",
        "thermal_chi_squared", "thermal_ndof",
    ]]
    df.attrs = {"nu_unit": df_minimum.attrs["nu_unit"]}
    return df


def main() -> None:
    args = parse_args()
    inputs = GeneralInputs.from_yaml(args.input)
    df_minimum = FileReader.table_from_csv(args.minimum_chi_squared)
    df = build_parameter_table(inputs, df_minimum)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    FileWriter.df_to_csv(
        path=args.output,
        df=df,
    )


if __name__ == "__main__":
    main()
