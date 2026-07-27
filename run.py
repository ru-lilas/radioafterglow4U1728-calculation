from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd


RUNS_DIR = Path("data")
INDEX_PATH = RUNS_DIR / "index.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="指定したrun_idの計算を実行します。",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="実行するrun_id。省略時は対話的に入力します。",
    )
    return parser.parse_args()


def read_run_index(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"run indexが見つかりません: {path}"
        )

    df = pd.read_csv(
        path,
        dtype={
            "run_id": "int64",
        },
    )

    required_columns = {
        "run_id",
        "run_name",
        "input_path",
    }
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"{path}に必要な列がありません: {missing}"
        )

    if df["run_id"].duplicated().any():
        duplicated_ids = (
            df.loc[df["run_id"].duplicated(keep=False), "run_id"]
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            f"run_idが重複しています: {duplicated_ids}"
        )

    return df


def show_runs(df: pd.DataFrame) -> None:
    display_columns = [
        column
        for column in [
            "run_id",
            "run_name",
            "description",
            "free_parameters",
            "status",
            "created_at",
        ]
        if column in df.columns
    ]

    print("\nAvailable runs")
    print("-" * 80)
    print(
        df[display_columns]
        .sort_values("run_id")
        .to_string(index=False)
    )
    print()


def ask_run_id(df: pd.DataFrame) -> int:
    valid_ids = set(df["run_id"].tolist())

    while True:
        value = input("Run ID > ").strip()

        if value.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt

        try:
            run_id = int(value)
        except ValueError:
            print("run_idは整数で入力してください。")
            continue

        if run_id not in valid_ids:
            print(
                f"run_id={run_id}は存在しません。"
            )
            continue

        return run_id


def select_run(
    df: pd.DataFrame,
    run_id: int,
) -> pd.Series:
    selected = df.loc[df["run_id"] == run_id]

    if selected.empty:
        raise ValueError(
            f"run_id={run_id}はindex.csvに存在しません。"
        )

    return selected.iloc[0]


def resolve_input_path(
    row: pd.Series,
    *,
    project_root: Path,
) -> Path:
    input_path = Path(str(row["input_path"]))

    if not input_path.is_absolute():
        input_path = project_root / input_path

    input_path = input_path.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(
            f"入力YAMLが見つかりません: {input_path}"
        )

    return input_path


def run_calculation(
    *,
    run_id: int,
    input_path: Path,
) -> None:
    run_id_fmt = f"{run_id:06d}"
    run_dir = f"run_{run_id_fmt}"
    command = [
        "make",
        f"RUN_DIR={run_dir}",
        f"data/{run_dir}/parameter_table.csv",
    ]

    print("\n実行コマンド:")
    print(" ".join(command))
    print()

    subprocess.run(
        command,
        check=True,
    )


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parent
    index_path = project_root / INDEX_PATH

    df_index = read_run_index(index_path)

    if args.run_id is None:
        show_runs(df_index)
        run_id = ask_run_id(df_index)
    else:
        run_id = args.run_id

    row = select_run(
        df_index,
        run_id,
    )

    input_path = resolve_input_path(
        row,
        project_root=project_root,
    )

    print(f"run_id    : {run_id}")
    print(f"run_name  : {row['run_name']}")
    print(f"input_path: {input_path}")

    run_calculation(
        run_id=run_id,
        input_path=input_path,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n実行を中止しました。")
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"計算コマンドが終了コード"
            f"{error.returncode}で失敗しました。"
        ) from error
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
