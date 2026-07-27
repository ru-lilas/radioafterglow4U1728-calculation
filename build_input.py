from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

TEMPLATE_PATH = Path("input/general.yaml")
RUNS_DIR = Path("data")
INDEX_PATH = RUNS_DIR / "index.csv"

yaml = YAML()
yaml.preserve_quotes = True
yaml.default_flow_style = False


def load_yaml(path: Path) -> CommentedMap:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.load(file)

    if not isinstance(config, CommentedMap):
        raise TypeError(
            f"YAML root must be a mapping, got {type(config).__name__}"
        )

    return config

def print_yaml(config: CommentedMap) -> None:
    import sys

    yaml.dump(config, sys.stdout)


def get_nested_value(
    config: CommentedMap,
    key_path: str,
) -> Any:
    keys = key_path.split(".")
    current: Any = config

    for key in keys:
        if not isinstance(current, dict):
            raise KeyError(
                f"{key!r} cannot be accessed because "
                "its parent is not a mapping"
            )

        if key not in current:
            raise KeyError(f"Key not found: {key_path}")

        current = current[key]

    return current


def set_nested_value(
    config: CommentedMap,
    key_path: str,
    value: Any,
) -> None:
    keys = key_path.split(".")
    current: Any = config

    for key in keys[:-1]:
        if not isinstance(current, dict):
            raise KeyError(
                f"{key!r} cannot be accessed because "
                "its parent is not a mapping"
            )

        if key not in current:
            raise KeyError(f"Key not found: {key_path}")

        current = current[key]

    final_key = keys[-1]

    if not isinstance(current, dict):
        raise KeyError(
            f"Cannot set {final_key!r}: parent is not a mapping"
        )

    if final_key not in current:
        raise KeyError(f"Key not found: {key_path}")

    current[final_key] = value


def parse_value(text: str) -> Any:
    """
    入力文字列をYAMLとして解釈する。

    例:
        0.4          -> float
        256          -> int
        true         -> bool
        "[a, b]"     -> list
        "GHz"        -> str
    """
    return yaml.load(text)


def edit_config(config: CommentedMap) -> None:
    while True:
        print()
        print_yaml(config)
        print()
        print("操作:")
        print("  e: 値を変更")
        print("  y: この設定でrunを生成")
        print("  q: 終了")

        command = input("> ").strip().lower()

        if command == "e":
            edit_one_value(config)

        elif command == "y":
            return

        elif command == "q":
            raise KeyboardInterrupt

        else:
            print(f"Unknown command: {command!r}")


def edit_one_value(config: CommentedMap) -> None:
    key_path = input(
        "変更するキーをドット区切りで入力してください: "
    ).strip()

    try:
        current_value = get_nested_value(config, key_path)
    except KeyError as error:
        print(error)
        return

    print(f"現在値: {current_value!r}")

    raw_value = input("新しい値: ").strip()

    try:
        new_value = parse_value(raw_value)
    except Exception as error:
        print(f"値を解釈できませんでした: {error}")
        return

    print(f"変更後: {new_value!r}")

    confirm = input("この変更を適用しますか？ [y/N]: ").strip().lower()

    if confirm != "y":
        print("変更を取り消しました")
        return

    set_nested_value(
        config=config,
        key_path=key_path,
        value=new_value,
    )


def normalize_for_hash(value: Any) -> Any:
    """
    ruamel.yaml固有の型をJSONへ変換可能な通常型へ直す。
    """
    if isinstance(value, dict):
        return {
            str(key): normalize_for_hash(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            normalize_for_hash(item)
            for item in value
        ]

    return value


def calculate_config_hash(
    config: CommentedMap,
    *,
    length: int = 12,
) -> str:
    normalized = normalize_for_hash(config)

    serialized = json.dumps(
        normalized,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    digest = hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()

    return digest[:length]


def read_existing_run_ids(index_path: Path) -> list[int]:
    if not index_path.exists():
        return []

    with index_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        return [
            int(row["run_id"])
            for row in reader
        ]


def generate_next_run_id(index_path: Path) -> int:
    existing_ids = read_existing_run_ids(index_path)

    if not existing_ids:
        return 1

    return max(existing_ids) + 1


def write_config(
    config: CommentedMap,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=False,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.dump(config, file)


def extract_index_values(
    config: CommentedMap,
) -> dict[str, Any]:
    free_parameters = config["chi2fitting"]["free_parameters"]

    return {
        "n_free_parameters": len(free_parameters),
        "free_parameters": "|".join(
            str(parameter)
            for parameter in free_parameters
        ),
        "eps_th": config["physical_parameters"]["eps_th"],
        "eps_b": config["physical_parameters"]["eps_b"],
        "time_min": (
            config["chi2fitting"]
            ["obs_sampling"]
            ["timewindow"]
            ["min"]
        ),
        "time_max": (
            config["chi2fitting"]
            ["obs_sampling"]
            ["timewindow"]
            ["max"]
        ),
    }


def append_index(
    *,
    index_path: Path,
    run_id: int,
    config_hash: str,
    input_path: Path,
    config: CommentedMap,
) -> None:
    index_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    extracted = extract_index_values(config)

    row = {
        "run_id": run_id,
        "run_name": f"run_{run_id:06d}",
        "config_hash": config_hash,
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "input_path": input_path.as_posix(),
        **extracted,
    }

    fieldnames = list(row.keys())
    file_exists = index_path.exists()

    with index_path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def find_duplicate_hash(
    index_path: Path,
    config_hash: str,
) -> str | None:
    if not index_path.exists():
        return None

    with index_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["config_hash"] == config_hash:
                return row["run_name"]

    return None


def create_run(config: CommentedMap) -> Path:
    config_hash = calculate_config_hash(config)

    duplicate_run = find_duplicate_hash(
        INDEX_PATH,
        config_hash,
    )

    if duplicate_run is not None:
        print(
            "同じ設定がすでに存在します: "
            f"{duplicate_run}"
        )

        confirm = input(
            "重複したrunを生成しますか？ [y/N]: "
        ).strip().lower()

        if confirm != "y":
            raise RuntimeError("run生成を中止しました")

    run_id = generate_next_run_id(INDEX_PATH)
    run_name = f"run_{run_id:06d}"

    run_dir = RUNS_DIR / run_name
    input_path = run_dir / "input.yaml"

    write_config(
        config=config,
        output_path=input_path,
    )

    append_index(
        index_path=INDEX_PATH,
        run_id=run_id,
        config_hash=config_hash,
        input_path=input_path,
        config=config,
    )

    return run_dir


def main() -> None:
    config = load_yaml(TEMPLATE_PATH)

    try:
        edit_config(config)
    except KeyboardInterrupt:
        print()
        print("終了しました")
        return

    print()
    print("最終設定:")
    print_yaml(config)

    confirm = input(
        "この設定でディレクトリを生成しますか？ [y/N]: "
    ).strip().lower()

    if confirm != "y":
        print("生成を中止しました")
        return

    try:
        run_dir = create_run(config)
    except RuntimeError as error:
        print(error)
        return

    print()
    print(f"生成しました: {run_dir}")


if __name__ == "__main__":
    main()
