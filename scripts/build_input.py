from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from decimal import Decimal
from tempfile import NamedTemporaryFile
from copy import deepcopy
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

    seen: set[int] = set()

    def preserve_anchors(value: Any) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        anchor = getattr(value, "anchor", None)
        if anchor is not None and anchor.value:
            value.yaml_set_anchor(anchor.value, always_dump=True)
        if isinstance(value, dict):
            for item in value.values():
                preserve_anchors(item)
        elif isinstance(value, list):
            for item in value:
                preserve_anchors(item)

    preserve_anchors(config)
    return config


@dataclass(frozen=True, slots=True)
class ConfigRow:
    group: str
    name: str
    key_path: str
    value: Any


def build_config_rows(config: CommentedMap) -> list[ConfigRow]:
    """Group grid settings, values, units and related scalar parameters into rows"""
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
    """List editable leaves while treating frequency lists as a single value"""
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
    # 不正なパスでは、継承値のコピーも含めて設定を変更しない。
    get_nested_value(config, key_path)
    keys = key_path.split(".")
    parent: Any = config
    for key in keys[:-1]:
        if isinstance(parent, CommentedMap):
            explicit_keys = {name for name, _ in parent.non_merged_items()}
            if key not in explicit_keys:
                # <<で継承したマッピングを編集するときは局所的な上書きにする。
                parent[key] = deepcopy(parent[key])
        parent = parent[key]
    parent[keys[-1]] = value


def parse_value(text: str) -> Any:
    """
    Parse an input string as YAML

    Examples:
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
    Convert YAML containers into plain JSON-compatible containers
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
    """Extract index settings while preserving logarithmic grid bounds"""
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
        "eps_th": "model_parameters.shock.common.eps_th",
        "eps_b": "model_parameters.shock.common.eps_b",
        "mu": "model_parameters.shock.common.molecular_weight.mu",
        "mu_e": "model_parameters.shock.common.molecular_weight.mu_e",
        "distance": "model_parameters.distance.value",
        "distance_unit": "model_parameters.distance.unit",
        "p_index": "model_parameters.shock.powerlaw.p_index",
        "f_e": "model_parameters.shock.powerlaw.f_e",
        "thermal_only": "model_parameters.switches.thermal_only",
        "exact_dist_factors": "model_parameters.switches.exact_dist_factors",
        "observation_frequencies": "candidate_models_selection.observation_frequencies.values",
        "nu_unit": "candidate_models_selection.observation_frequencies.unit",
        "time_min": "chi_squared.timewindow.min",
        "time_max": "chi_squared.timewindow.max",
        "time_unit": "chi_squared.timewindow.unit",
        "burster_name": "burster_property.burster_name",
        "p_indices_start": "powerlaw_sensitivity.p_indices.start",
        "p_indices_stop": "powerlaw_sensitivity.p_indices.stop",
        "p_indices_num": "powerlaw_sensitivity.p_indices.num",
        "p_indices_scale": "powerlaw_sensitivity.p_indices.scale",
        "f_es_start": "powerlaw_sensitivity.f_es.start",
        "f_es_stop": "powerlaw_sensitivity.f_es.stop",
        "f_es_num": "powerlaw_sensitivity.f_es.num",
        "f_es_scale": "powerlaw_sensitivity.f_es.scale",
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
    """Reject incompatible index headers before appending rows"""
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

    temporary: Path | None = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=index_path.parent,
                                prefix=".index_", suffix=".csv", delete=False) as file:
            temporary = Path(file.name)
            if file_exists:
                with index_path.open(encoding="utf-8", newline="") as existing:
                    file.write(existing.read())
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(index_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


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
    """Apply KEY=VALUE overrides in order, parsing each value as YAML"""
    for override in overrides:
        key, separator, raw_value = override.partition("=")
        if not separator or not key.strip() or not raw_value.strip():
            raise ValueError(f"--setはKEY=VALUE形式で指定してください: {override!r}")
        set_nested_value(config, key.strip(), parse_value(raw_value))



EPS_B_PATH = "model_parameters.shock.common.eps_b"
SWEEP_FIELDS = ["eps_b", "run_id", "run_name", "input_path", "config_hash", "status"]


def validate_eps_b_values(values: list[float]) -> list[float]:
    """Validate and deduplicate the sweep values in input order"""
    if not values:
        raise ValueError("eps_Bのリストが空です")
    unique: list[float] = []
    for value in values:
        if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"eps_Bには0より大きく1以下の有限値が必要です: {value}")
        if value in unique:
            print(f"Duplicate eps_B skipped: {value:g}")
        else:
            unique.append(value)
    return unique


def semantic_config(value: Any) -> Any:
    """Compare effective settings independently of YAML formatting and numeric spelling"""
    if isinstance(value, dict):
        return {str(key): semantic_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(semantic_config(item) for item in value)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)):
        return ("number", str(Decimal(str(value)).normalize()))
    return (type(value).__name__, value)


def sweep_family(config: CommentedMap) -> Any:
    """Compare all effective settings except eps_B"""
    plain = normalize_for_hash(config)
    parent = plain
    keys = EPS_B_PATH.split(".")
    for key in keys[:-1]:
        parent = parent[key]
    parent[keys[-1]] = None
    return semantic_config(plain)



def make_anchor_names_unique(config: CommentedMap) -> None:
    """Keep detached aliases valid by assigning unique names to distinct anchors"""
    seen: set[int] = set()
    used: set[str] = set()

    def visit(value: Any) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        anchor = getattr(value, "anchor", None)
        if anchor is not None and anchor.value:
            name = anchor.value
            if name in used:
                suffix = 1
                while f"{name}_sweep{suffix}" in used:
                    suffix += 1
                name = f"{name}_sweep{suffix}"
                value.yaml_set_anchor(name, always_dump=True)
            used.add(name)
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
            for merged in getattr(value, "merge", ()):
                visit(merged)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(config)

def build_sweep_config(base: CommentedMap, eps_b: float) -> CommentedMap:
    """Build an independent configuration changing only the designated eps_B field"""
    config = deepcopy(base)
    # Detach each parent mapping so aliases outside this path retain their values
    parent = config
    for key in EPS_B_PATH.split(".")[:-1]:
        parent[key] = deepcopy(parent[key])
        parent = parent[key]
    set_nested_value(config, EPS_B_PATH, eps_b)
    make_anchor_names_unique(config)
    if sweep_family(config) != sweep_family(base):
        raise ValueError("eps_B以外の設定が変わっています")
    return config


def indexed_input_path(row: dict[str, str], runs_dir: Path) -> Path:
    """Resolve indexed inputs using the existing path convention"""
    path = Path(row["input_path"])
    if path.exists():
        return path
    # Allow an index tree to be relocated as a whole
    relocated = runs_dir / row["run_name"] / "input.yaml"
    if relocated.exists():
        return relocated
    raise FileNotFoundError(f"Indexed input is missing: {path}")


def read_verified_runs(runs_dir: Path) -> list[tuple[dict[str, str], CommentedMap, Path]]:
    """Read existing inputs and verify their recorded fingerprints"""
    index_path = runs_dir / "index.csv"
    if not index_path.exists() or index_path.stat().st_size == 0:
        return []
    with index_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    verified = []
    for row in rows:
        path = indexed_input_path(row, runs_dir)
        config = load_yaml(path)
        if calculate_config_hash(config) != row["config_hash"]:
            raise ValueError(f"Indexed configuration hash disagrees with input: {path}")
        verified.append((row, config, path))
    return verified


def validate_sweep_manifest(path: Path, configs: list[CommentedMap], *,
                            allow_partial: bool = False) -> list[dict[str, str]]:
    """Allow reruns of the same sweep without replacing another sweep"""
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != SWEEP_FIELDS:
            raise ValueError(f"Sweep manifest columns differ: {path}")
        rows = list(reader)
    if len(rows) > len(configs) or (not allow_partial and len(rows) != len(configs)):
        raise ValueError(f"Sweep manifest belongs to another sweep: {path}")
    for row, config in zip(rows, configs):
        existing_path = Path(row["input_path"])
        if not existing_path.exists():
            raise FileNotFoundError(f"Sweep input is missing: {existing_path}")
        existing_config = load_yaml(existing_path)
        if (float(row["eps_b"]) != get_nested_value(config, EPS_B_PATH)
                or semantic_config(existing_config) != semantic_config(config)):
            raise ValueError(f"Sweep manifest belongs to another sweep: {path}")
        if calculate_config_hash(existing_config) != row["config_hash"]:
            raise ValueError(f"Sweep manifest hash disagrees with input: {existing_path}")
        if row["status"] not in ("created", "reused"):
            raise ValueError(f"Invalid sweep status: {row['status']}")
    return rows


def write_sweep_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    """Replace the manifest atomically with the completed sweep records"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                prefix=".sweep_", suffix=".csv", delete=False) as file:
            temporary = Path(file.name)
            writer = csv.DictWriter(file, fieldnames=SWEEP_FIELDS)
            writer.writeheader()
            writer.writerows(records)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()



def sweep_plan(base: CommentedMap, values: list[float]) -> dict[str, object]:
    """Record the full requested sweep independently of completion status"""
    return {"eps_b_values": values,
            "family": json.loads(json.dumps(sweep_family(base), sort_keys=True, default=str))}


@dataclass(frozen=True, slots=True)
class SavedSweepPlan:
    eps_b_values: list[float]
    family: object

    def to_dict(self) -> dict[str, object]:
        """Return the serialized sweep-plan fields"""
        return {"eps_b_values": self.eps_b_values, "family": self.family}


def read_sweep_plan(path: Path) -> SavedSweepPlan:
    """Validate JSON fields and return a typed saved sweep plan"""
    with path.open(encoding="utf-8") as file:
        raw: object = json.load(file)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid sweep plan: {path}")
    fields: dict[str, object] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ValueError(f"Invalid sweep-plan key: {path}")
        fields[key] = value
    raw_values = fields.get("eps_b_values")
    family = fields.get("family")
    if not isinstance(raw_values, list) or not isinstance(family, dict):
        raise ValueError(f"Invalid sweep plan fields: {path}")
    values: list[float] = []
    for item in raw_values:
        value: object = item
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Invalid eps_B value in saved sweep plan: {path}")
        values.append(float(value))
    planned = validate_eps_b_values(values)
    if len(planned) != len(values):
        raise ValueError(f"Duplicate values in saved sweep plan: {path}")
    return SavedSweepPlan(eps_b_values=planned, family=family)


def write_sweep_plan(path: Path, plan: dict[str, object]) -> None:
    """Write sweep identity atomically before the first RUN is generated"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix=".sweep_plan_", delete=False) as file:
            temporary = Path(file.name)
            json.dump(plan, file, sort_keys=True, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

def generate_eps_b_sweep(base: CommentedMap, values: list[float], *,
                         runs_dir: Path, sweep_index: Path,
                         extend_sweep: bool = False) -> list[dict[str, Any]]:
    """Generate, reuse or extend a sweep while preserving completed RUN assignments"""
    requested = validate_eps_b_values(values)
    index_path = runs_dir / "index.csv"
    if sweep_index.resolve() == index_path.resolve():
        raise ValueError("sweep-indexはRUNのindex.csvとは別のパスにしてください")
    plan_path = sweep_index.with_suffix(".plan.json")
    existing_plan = plan_path.exists()
    saved_plan: SavedSweepPlan | None = None
    if existing_plan:
        saved_plan = read_sweep_plan(plan_path)
    if extend_sweep:
        if saved_plan is None or not sweep_index.is_file():
            raise FileNotFoundError("--extend-sweepには既存の走査対応表と.plan.jsonが必要です")
        if saved_plan.family != sweep_plan(base, requested)["family"]:
            raise ValueError(f"Common sweep settings differ: {plan_path}")
        planned: list[float] = saved_plan.eps_b_values
        values = planned.copy()
        for value in requested:
            if value in values:
                print(f"Already planned eps_B skipped: {value:g}")
            else:
                values.append(value)
        print(f"New eps_B values added: {len(values) - len(planned)}")
    else:
        values = requested
    plan = sweep_plan(base, values)
    if saved_plan is not None and not extend_sweep and saved_plan.to_dict() != plan:
        raise ValueError(f"Sweep plan belongs to another sweep: {plan_path}")
    configs = [build_sweep_config(base, value) for value in values]
    for config in configs:
        extracted = extract_index_values(config)
        validate_index_header(index_path, [
            "run_id", "run_name", "config_hash", "created_at", "input_path", *extracted,
        ])
    verified = read_verified_runs(runs_dir)
    completed = validate_sweep_manifest(sweep_index, configs, allow_partial=existing_plan)
    records: list[dict[str, Any]] = []
    used_ids: set[int] = set()
    for row in completed:
        run_id = int(row["run_id"])
        expected_path = (runs_dir / row["run_name"] / "input.yaml").resolve()
        input_path = Path(row["input_path"])
        if input_path.resolve() != expected_path:
            raise ValueError(f"Sweep entry is outside the requested RUN root: {input_path}")
        indexed = [(item, old, path) for item, old, path in verified
                   if int(item["run_id"]) == run_id and item["run_name"] == row["run_name"]
                   and path.resolve() == input_path.resolve()]
        if len(indexed) != 1 or run_id in used_ids:
            raise ValueError(f"Existing sweep RUN is not uniquely registered in {index_path}: {row['run_name']}")
        used_ids.add(run_id)
        records.append(dict(eps_b=float(row["eps_b"]), run_id=run_id,
                            run_name=row["run_name"], input_path=row["input_path"],
                            config_hash=row["config_hash"], status="reused"))
    # Check all required new directories before changing either plan or manifest
    next_id = generate_next_run_id(index_path)
    pending = configs[len(records):]
    new_count = sum(not any(semantic_config(old) == semantic_config(config)
                           for _, old, _ in verified) for config in pending)
    for run_id in range(next_id, next_id + new_count):
        if (runs_dir / f"run_{run_id:06d}").exists():
            raise FileExistsError(f"RUN directory is already occupied: run_{run_id:06d}")
    if saved_plan is None or saved_plan.to_dict() != plan:
        write_sweep_plan(plan_path, plan)
    # Never shorten a completed manifest while replaying an existing sweep
    if records:
        write_sweep_manifest(sweep_index, records)
        for record in records:
            print(f"eps_B={record['eps_b']:g}: reused {record['run_name']}")
    for value, config in zip(values[len(records):], pending):
        try:
            matches = [(row, path) for row, old, path in verified
                       if semantic_config(old) == semantic_config(config)]
            if matches:
                matches.sort(key=lambda item: int(item[0]["run_id"]))
                row, input_path = matches[0]
                if len(matches) > 1:
                    print("Equivalent RUNs: " + ", ".join(item[0]["run_name"] for item in matches))
                record = dict(eps_b=value, run_id=int(row["run_id"]), run_name=row["run_name"],
                              input_path=input_path.as_posix(), config_hash=row["config_hash"], status="reused")
            else:
                run_dir = create_run(config, runs_dir=runs_dir, interactive=False)
                input_path = run_dir / "input.yaml"
                record = dict(eps_b=value, run_id=int(run_dir.name.removeprefix("run_")),
                              run_name=run_dir.name, input_path=input_path.as_posix(),
                              config_hash=calculate_config_hash(config), status="created")
            records.append(record)
            write_sweep_manifest(sweep_index, records)
            print(f"eps_B={value:g}: {record['status']} {record['run_name']}")
        except (OSError, KeyError, TypeError, ValueError, RuntimeError, YAMLError) as error:
            raise RuntimeError(f"eps_B={value:g}, RUN root={runs_dir}, manifest={sweep_index}: {error}") from error
    return records

def main() -> None:
    parser = argparse.ArgumentParser(
        description="general.yamlを基にrunのinput.yamlとindex.csvを生成する。"
    )
    parser.add_argument("--mode", choices=("single", "eps-b-sweep"), default="single")
    parser.add_argument("--eps-b-values", type=float, nargs="+")
    parser.add_argument("--sweep-index", type=Path)
    parser.add_argument("--extend-sweep", action="store_true",
                        help="既存の走査条件を維持し、指定したeps_B値を末尾に追加")
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
    if args.mode == "single" and (args.eps_b_values is not None or args.sweep_index is not None or args.extend_sweep):
        parser.error("走査専用引数には--mode eps-b-sweepが必要です")
    if args.mode == "eps-b-sweep":
        if args.eps_b_values is None or args.sweep_index is None:
            parser.error("eps-b-sweepには--eps-b-valuesと--sweep-indexが必要です")
        if args.allow_duplicate:
            parser.error("eps-b-sweepでは--allow-duplicateを使用できません")

    try:
        config = load_yaml(args.template)
        apply_overrides(config, args.overrides)
        if not args.yes:
            edit_config(config)
        if args.mode == "eps-b-sweep":
            eps_b_values: list[float] | None = args.eps_b_values
            sweep_index: Path | None = args.sweep_index
            if eps_b_values is None or sweep_index is None:
                parser.error("eps-b-sweepには--eps-b-valuesと--sweep-indexが必要です")
            generate_eps_b_sweep(config, eps_b_values, runs_dir=args.runs_dir,
                                 sweep_index=sweep_index, extend_sweep=args.extend_sweep)
            print(f"Sweep manifest: {args.sweep_index}")
            return
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
