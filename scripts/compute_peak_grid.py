"""Compute a synchrotron peak grid in memory-budgeted model batches"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from module.inputs_as_dataclass import GeneralInputs
from module.memory_utils import GIB, check_memory_budget, estimate_array_memory_bytes
from module.models import ModelParameters, SynchrotronResults
from module.synchrotron_numerical_table import ThermalSynchrotronIntegrals
from module.utils import FileReader, FileWriter
from module.timing_utils import timed

# Model-specific allowance for component arrays and temporary arrays.
ESTIMATED_FLOAT64_ARRAYS = 32


def model_batch_size(n_models: int, n_xm: int, max_memory_gib: float) -> int:
    """Budget dense spectral arrays; input tables and peak rows are additional"""
    if n_models < 1 or n_xm < 1:
        raise ValueError("モデル数とxm点数は1以上が必要です")
    per_model = estimate_array_memory_bytes(
        shape=(1, n_xm), array_count=ESTIMATED_FLOAT64_ARRAYS,
    )
    check_memory_budget(per_model, max_memory_gib)
    return min(n_models, int(max_memory_gib * GIB / per_model))


def compute_peak_table(
    model_params: ModelParameters,
    xm: NDArray[np.float64],
    table: ThermalSynchrotronIntegrals,
    *,
    max_memory_gib: float,
) -> pd.DataFrame:
    """Retain only peak rows, preserving original model IDs and their order"""
    if xm.ndim != 2 or xm.shape[0] != 1:
        raise ValueError("xmは(1, M)の配列が必要です")
    indices = model_params.model_indices.ravel()
    batch_size = model_batch_size(indices.size, xm.shape[1], max_memory_gib)
    estimated_bytes = estimate_array_memory_bytes(
        shape=(batch_size, xm.shape[1]), array_count=ESTIMATED_FLOAT64_ARRAYS,
    )
    n_batches = (indices.size + batch_size - 1) // batch_size
    print(
        f"Models: {indices.size}; xm points: {xm.shape[1]}; "
        f"batch size: {batch_size}; batches: {n_batches}; "
        f"estimated batch arrays: {estimated_bytes / GIB:.2f} GiB",
        flush=True,
    )
    peak_tables = []
    for batch_number, start in enumerate(range(0, indices.size, batch_size), start=1):
        batch = model_params.select_models(indices[start:start + batch_size].tolist())
        results = SynchrotronResults.compute_from(model_params=batch, xm=xm, table=table)
        peak_tables.append(results.peak().to_df())
        del results, batch
        print(f"Peak batch {batch_number}/{n_batches} completed", flush=True)
    peaks = pd.concat(peak_tables, ignore_index=True)
    peaks.attrs = peak_tables[0].attrs.copy()
    return peaks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "integral_table", "model_parameter_table", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--max_memory_gib",
        type=float,
        default=4.0,
        help="計算配列の推定メモリ上限（GiB、既定値: 4.0）",
    )
    return parser.parse_args()


@timed
def main() -> None:
    args = parse_args()
    print("Computing peak_grid...", flush=True)

    conf_lc = GeneralInputs.from_yaml(args.input).model_lightcurves
    table = ThermalSynchrotronIntegrals.from_csv(args.integral_table)
    df_model = FileReader.table_from_csv(args.model_parameter_table)
    model_params = ModelParameters.from_df(df_model)
    xm = conf_lc.xm.to_floatvector().values[None, :]

    peaks = compute_peak_table(
        model_params=model_params,
        xm=xm,
        table=table,
        max_memory_gib=args.max_memory_gib,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    FileWriter.df_to_csv(
        path=args.output,
        df=peaks,
    )
    print(f"Saved peak_grid: {args.output}", flush=True)


if __name__ == "__main__":
    main()
