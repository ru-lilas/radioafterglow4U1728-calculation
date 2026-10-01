from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.error import YAMLError

TEMPLATE_PATH = Path("input/general.yaml")
RUNS_DIR = Path("data")

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

@dataclass(frozen=True, slots=True)
class ConfigRow:
    group: str
    name: str
    key_path: str
    value: Any

def build_config_rows(config: CommentedMap) -> list[ConfigRow]:
    """格子設定・値と単位・関連するスカラーを1行にまとめる。"""
    rows: list[ConfigRow] = []

    def add(group: str, name: str, path: str, value: Any) -> None:
        if isinstance(value, dict):
            contents = value.get("values", value)
            if isinstance(contents, dict) and any(
                isinstance(item, dict) for item in contents.values()
            ):
                for key, item in value.items():
                    add(group, f"{name}.{key}", f"{path}.{key}", item)
                return
        rows.append(ConfigRow(group, name, path, value))

    # モデルを先頭に置き、共通の単位定義は最後に表示する。
    groups = [key for key in config if key != "units"]
    if "units" in config:
        groups.append("units")
    for group in groups:
        values = config[group]
        if isinstance(values, dict):
            for name, value in values.items():
                add(str(group), str(name), f"{group}.{name}", value)
        else:
            add(str(group), str(group), str(group), values)
    return rows

def format_table_value(value: Any) -> tuple[str, str]:
    unit = "—"
    if isinstance(value, dict):
        unit = str(value.get("unit", "—"))
        if "values" in value:
            value = value["values"]
        elif "value" in value:
            value = value["value"]
        else:
            value = {key: item for key, item in value.items() if key != "unit"}
    if isinstance(value, dict):
        text = ", ".join(f"{key}={item}" for key, item in value.items())
    elif isinstance(value, list):
        text = json.dumps(normalize_for_hash(value), ensure_ascii=False)
    else:
        text = str(value)
    return text, unit


def print_config_table(config: CommentedMap) -> list[ConfigRow]:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    rows = build_config_rows(config)
    table = Table(title="runの設定", show_lines=False)
    table.add_column("番号", justify="right", no_wrap=True)
    table.add_column("グループ", style="cyan")
    table.add_column("パラメータ")
    table.add_column("現在値", ratio=2)
    table.add_column("単位", no_wrap=True)
    for i, row in enumerate(rows):
        value, unit = format_table_value(row.value)
        group = row.group if i == 0 or rows[i - 1].group != row.group else ""
        end_section = i == len(rows) - 1 or rows[i + 1].group != row.group
        table.add_row(
            str(i + 1), Text(group), Text(row.name), Text(value), Text(unit),
            end_section=end_section,
        )
    Console().print(table)
    return rows

def editable_paths(value: Any, path: str) -> list[str]:
    """複合パラメータの編集先を列挙。周波数のリストはまとめて編集する。"""
    if isinstance(value, dict):
        return [
            leaf
            for key, item in value.items()
            for leaf in editable_paths(item, f"{path}.{key}")
        ]
    return [path]

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
    parent_path, separator, final_key = key_path.rpartition(".")
    parent = get_nested_value(config, parent_path) if separator else config
    if not isinstance(parent, dict):
        raise KeyError(f"Cannot set {key_path!r}: parent is not a mapping")
    if final_key not in parent:
        raise KeyError(f"Key not found: {key_path}")
    parent[final_key] = value


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
        rows = print_config_table(config)
        print("操作: 番号で編集 / e: キー指定で編集 / y: runを生成 / q: 終了")
        command = input("> ").strip().lower()
        if command == "e" or command.isdecimal():
            edit_one_value(config, rows, command if command != "e" else None)
        elif command == "y":
            return
        elif command == "q":
            raise KeyboardInterrupt
        else:
            print(f"Unknown command: {command!r}")


def edit_one_value(
    config: CommentedMap,
    rows: list[ConfigRow],
    selection: str | None = None,
) -> None:
    selection = selection or input("変更する番号またはドット区切りのキー: ").strip()
    if selection.isdecimal():
        index = int(selection) - 1
        if not 0 <= index < len(rows):
            print("表示されている番号を指定してください")
            return
        row = rows[index]
        paths = editable_paths(row.value, row.key_path)
        if not paths:
            print("編集する値がありません")
            return
        if len(paths) == 1:
            key_path = paths[0]
        else:
            print(f"{row.key_path}の編集項目:")
            for i, path in enumerate(paths, start=1):
                name = path.removeprefix(row.key_path + ".")
                print(f"  {i}: {name} = {get_nested_value(config, path)!r}")
            field = input("変更する項目の番号（空欄で戻る）: ").strip()
            if not field:
                return
            if not field.isdecimal() or not 1 <= int(field) <= len(paths):
                print("表示されている番号を指定してください")
                return
            key_path = paths[int(field) - 1]
    else:
        key_path = selection

    try:
        current_value = get_nested_value(config, key_path)
    except KeyError as error:
        print(error)
        return
    print(f"{key_path}: {current_value!r}")
    raw_value = input("新しい値（空欄で戻る）: ").strip()
    if not raw_value:
        return
    try:
        new_value = parse_value(raw_value)
    except YAMLError as error:
        print(f"値を解釈できませんでした: {error}")
        return
    print(f"変更後: {new_value!r}")
    if input("この変更を適用しますか？ [y/N]: ").strip().lower() != "y":
        print("変更を取り消しました")
        return
    set_nested_value(config, key_path, new_value)


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
    """general.yamlの主要設定を索引へ記録する（logのstart/stopもそのまま）。"""
    paths = {
        "phi_th_start": "model_parameters.phi_th.values.start",
        "phi_th_stop": "model_parameters.phi_th.values.stop",
        "phi_th_num": "model_parameters.phi_th.values.num",
        "phi_th_scale": "model_parameters.phi_th.values.scale",
        "phi_th_unit": "model_parameters.phi_th.unit",
        "theta0_start": "model_parameters.theta0.start",
        "theta0_stop": "model_parameters.theta0.stop",
        "theta0_num": "model_parameters.theta0.num",
        "theta0_scale": "model_parameters.theta0.scale",
        "eps_th": "model_parameters.eps_th",
        "eps_b": "model_parameters.eps_b",
        "mu": "model_parameters.molecular_weight.mu",
        "mu_e": "model_parameters.molecular_weight.mu_e",
        "distance": "model_parameters.distance.value",
        "distance_unit": "model_parameters.distance.unit",
        "p_index": "model_parameters.powerlaw.p_index",
        "eps_e": "model_parameters.powerlaw.eps_e",
        "observation_frequencies": "candidate_models_selection.observation_frequencies.values",
        "nu_unit": "candidate_models_selection.observation_frequencies.unit",
        "time_min": "chi2.timewindow.min",
        "time_max": "chi2.timewindow.max",
        "time_unit": "chi2.timewindow.unit",
        "burster_name": "burster_property.burster_name",
    }
    values = {
        name: get_nested_value(config, path)
        for name, path in paths.items()
    }
    values["observation_frequencies"] = json.dumps(
        normalize_for_hash(values["observation_frequencies"]),
        ensure_ascii=False,
    )
    return values


def validate_index_header(index_path: Path, fieldnames: list[str]) -> None:
    """旧形式への追記で列がずれるのを防ぐ。"""
    if not index_path.exists() or index_path.stat().st_size == 0:
        return
    with index_path.open("r", encoding="utf-8", newline="") as file:
        header = next(csv.reader(file), [])
    if header != fieldnames:
        raise ValueError(
            f"{index_path}の列構成が現在の設定形式と異なります。"
            "別の保存先を--runs-dirで指定してください。"
        )


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
    validate_index_header(index_path, fieldnames)
    file_exists = index_path.exists() and index_path.stat().st_size > 0

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


def create_run(
    config: CommentedMap,
    *,
    runs_dir: Path = RUNS_DIR,
    allow_duplicate: bool = False,
    interactive: bool = True,
) -> Path:
    index_path = runs_dir / "index.csv"
    # 設定キーと索引の互換性をファイル生成前に確認する。
    extracted = extract_index_values(config)
    validate_index_header(index_path, [
        "run_id", "run_name", "config_hash", "created_at", "input_path",
        *extracted,
    ])
    config_hash = calculate_config_hash(config)

    duplicate_run = find_duplicate_hash(
        index_path,
        config_hash,
    )

    if duplicate_run is not None and not allow_duplicate:
        print(
            "同じ設定がすでに存在します: "
            f"{duplicate_run}"
        )

        if not interactive:
            raise RuntimeError("重複runを生成する場合は--allow-duplicateを指定してください")

        confirm = input(
            "重複したrunを生成しますか？ [y/N]: "
        ).strip().lower()

        if confirm != "y":
            raise RuntimeError("run生成を中止しました")

    run_id = generate_next_run_id(index_path)
    run_name = f"run_{run_id:06d}"

    run_dir = runs_dir / run_name
    input_path = run_dir / "input.yaml"

    write_config(
        config=config,
        output_path=input_path,
    )

    append_index(
        index_path=index_path,
        run_id=run_id,
        config_hash=config_hash,
        input_path=input_path,
        config=config,
    )

    return run_dir


def apply_overrides(config: CommentedMap, overrides: list[str]) -> None:
    """--set KEY=VALUEを順に適用する。VALUEはYAMLとして解釈する。"""
    for override in overrides:
        key, separator, raw_value = override.partition("=")
        if not separator or not key.strip() or not raw_value.strip():
            raise ValueError(f"--setはKEY=VALUE形式で指定してください: {override!r}")
        set_nested_value(config, key.strip(), parse_value(raw_value))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="general.yamlを基にrunのinput.yamlとindex.csvを生成する。"
    )
    parser.add_argument("--template", type=Path, default=TEMPLATE_PATH)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="ドット区切りのキーとYAML値を指定（複数指定可）",
    )
    parser.add_argument("--yes", action="store_true", help="対話操作を省略して生成")
    parser.add_argument(
        "--allow-duplicate", action="store_true", help="同じ設定のrun生成を許可"
    )
    args = parser.parse_args()

    try:
        config = load_yaml(args.template)
        apply_overrides(config, args.overrides)
        if not args.yes:
            edit_config(config)
        run_dir = create_run(
            config,
            runs_dir=args.runs_dir,
            allow_duplicate=args.allow_duplicate,
            interactive=not args.yes,
        )
    except (KeyboardInterrupt, EOFError):
        print("\n終了しました")
        return
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, YAMLError) as error:
        parser.exit(1, f"エラー: {error}\n")

    print(f"生成しました: {run_dir}")

if __name__ == "__main__":
    main()
