"""Compute a synchrotron peak grid after checking the array memory budget."""

import argparse
from pathlib import Path

from module.inputs_as_dataclass import GeneralInputs
from module.memory_utils import GIB, check_memory_budget, estimate_array_memory_bytes
from module.models import ModelParameters, SynchrotronResults
from module.synchrotron_numerical_table import ThermalSynchrotronIntegrals
from module.utils import FileReader, FileWriter

# Model-specific allowance for component arrays and temporary arrays.
ESTIMATED_FLOAT64_ARRAYS = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "integral_table", "model_parameter_table", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--max_memory_gib",
        type=float,
        default=6.0,
        help="計算配列の推定メモリ上限（GiB、既定値: 6.0）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Computing peak_grid...", flush=True)

    conf_lc = GeneralInputs.from_yaml(args.input).model_lightcurves
    table = ThermalSynchrotronIntegrals.from_csv(args.integral_table)
    df_model = FileReader.table_from_csv(args.model_parameter_table)
    model_params = ModelParameters.from_df(df_model)
    xm = conf_lc.xm.to_floatvector().values[None, :]

    estimated_bytes = estimate_array_memory_bytes(
        shape=(model_params.phi_th.size, xm.shape[1]),
        array_count=ESTIMATED_FLOAT64_ARRAYS,
    )
    check_memory_budget(estimated_bytes, args.max_memory_gib)
    print(f"Estimated array memory: {estimated_bytes / GIB:.2f} GiB", flush=True)

    results = SynchrotronResults.compute_from(
        model_params=model_params,
        xm=xm,
        table=table,
    )
    peaks = results.peak()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    FileWriter.df_to_csv(
        path=args.output,
        df=peaks.to_df(),
    )
    print(f"Saved peak_grid: {args.output}", flush=True)


if __name__ == "__main__":
    main()
