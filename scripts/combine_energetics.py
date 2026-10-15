"""Combine validated energetics tables from an eps_B or eps_th sweep"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal, TypeAlias
from numpy.typing import NDArray

import astropy.units as u
import numpy as np
import pandas as pd

from module.utils import FileReader, FileWriter

if __package__:
    from .build_input import SWEEP_SPECS, calculate_config_hash, get_nested_value, load_yaml, sweep_family
else:
    from build_input import SWEEP_SPECS, calculate_config_hash, get_nested_value, load_yaml, sweep_family

SweepParameter: TypeAlias = Literal["eps_b", "eps_th"]
FloatArray = NDArray[np.float64]
RTOL = 1e-10
ATOL = 0.0
REQUIRED = {
    "model_index", "nu", "mass_fraction", "energy_fraction", "kinetic_energy",
    "ejected_mass", "a_wind", "beta_sh", "t_peak", "eps_b", "eps_th", "mu_e_mu",
}
UNITS = {
    "kinetic_energy": ("energy_unit", "erg"),
    "ejected_mass": ("mass_unit", "g"),
    "a_wind": ("a_wind_unit", "g/cm"),
    "t_peak": ("t_unit", "min"),
    "nu": ("nu_unit", "GHz"),
}
IDENTIFIERS = {"run_id", "run_name", "config_hash", "result_hash_verified"}


@dataclass(frozen=True, slots=True)
class RunEntry:
    value: float
    run_id: int
    run_name: str
    input_path: Path
    config_hash: str | None
    parameter: SweepParameter = "eps_b"


def real_scalar(value: object) -> float:
    """Validate a finite real scalar before conversion"""
    if isinstance(value, np.generic):
        native: object = value.item()
        if isinstance(native, np.generic):
            raise TypeError("Unsupported NumPy scalar precision")
        return real_scalar(native)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Expected a real scalar, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Expected a finite number")
    return result


def read_manifest(path: Path) -> list[RunEntry]:
    """Read ordered RUN entries using the input generator's path convention"""
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        header = set(reader.fieldnames or [])
        axes = header & {"eps_b", "eps_th"}
        if len(axes) != 1:
            raise ValueError("Manifest must contain exactly one of eps_b and eps_th")
        parameter: SweepParameter = "eps_b" if "eps_b" in axes else "eps_th"
        required = {parameter, "run_id", "run_name", "input_path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing manifest columns: {sorted(required - set(reader.fieldnames or []))}")
        entries: list[RunEntry] = []
        ids: set[int] = set()
        paths: set[Path] = set()
        values: set[float] = set()
        names: set[str] = set()
        for row in reader:
            if any(not isinstance(row.get(key), str) or not row[key].strip() for key in required):
                raise ValueError(f"Missing manifest values: {row}")
            value = real_scalar(float(row[parameter]))
            run_id = int(row["run_id"])
            input_path = Path(row["input_path"])
            if not 0 < value <= 1 or run_id < 1 or not row["run_name"] or not row["input_path"]:
                raise ValueError(f"Invalid manifest entry: {row}")
            if run_id in ids or input_path.resolve() in paths or value in values or row["run_name"] in names:
                raise ValueError(f"Duplicate or ambiguous manifest entry: {row}")
            ids.add(run_id); paths.add(input_path.resolve()); values.add(value); names.add(row["run_name"])
            entries.append(RunEntry(value, run_id, row["run_name"], input_path, row.get("config_hash") or None, parameter))
    if not entries:
        raise ValueError("The sweep manifest is empty")
    return entries


def unit_factor(value: object, target: str) -> float:
    """Validate a unit string before computing its conversion factor"""
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing or invalid unit for conversion to {target}")
    return real_scalar(u.Unit(value).to(target))


def require_series_value(value: object, label: str) -> pd.Series:
    """Require a pandas operation to return a single Series"""
    if not isinstance(value, pd.Series):
        raise TypeError(f"Expected a Series for {label}, got {type(value).__name__}")
    return value


def require_series(df: pd.DataFrame, name: str) -> pd.Series:
    """Require a named column selection to return a single Series"""
    if name not in df:
        raise ValueError(f"Missing required column: {name}")
    column: object = df[name]
    return require_series_value(column, name)


def validate_column_labels(df: pd.DataFrame) -> None:
    """Reject duplicate column labels before processing a result table"""
    if not df.columns.is_unique:
        raise ValueError("Duplicate result column labels are not allowed")


def has_duplicate_results(df: pd.DataFrame) -> bool:
    """Return a Python boolean for duplicate result identities"""
    mask: object = df.duplicated(["run_id", "nu", "model_index"])
    return any(boolean_values(require_series_value(mask, "Duplicate result mask"), "Duplicate result mask"))


def numeric_column(df: pd.DataFrame, name: str) -> FloatArray:
    """Return a concrete finite real array after validating external scalars"""
    values: list[float] = []
    for raw in require_series(df, name).tolist():
        value: object = raw
        if isinstance(value, str):
            values.append(real_scalar(float(value)))
        else:
            values.append(real_scalar(value))
    return np.asarray(values, dtype=np.float64)


def boolean_values(column: pd.Series, label: str) -> list[bool]:
    """Validate every element before returning concrete Python booleans"""
    values: list[bool] = []
    for raw in column.tolist():
        value: object = raw
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{label} must contain only boolean values")
        values.append(bool(value))
    return values


def validate_run_names(df: pd.DataFrame, expected: str) -> None:
    """Validate optional RUN names as strings matching the manifest"""
    if "run_name" not in df:
        return
    names: list[str] = []
    for raw in require_series(df, "run_name").tolist():
        value: object = raw
        if not isinstance(value, str):
            raise TypeError("Result run_name must contain only strings")
        names.append(value)
    if not all(name == expected for name in names):
        raise ValueError("Result run_name disagrees with manifest")


def integer_column(df: pd.DataFrame, name: str) -> NDArray[np.int64]:
    """Preserve integer identifiers and reject fractional or invalid values"""
    values: list[int] = []
    for raw in require_series(df, name).tolist():
        value: object = raw
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool):
            raise TypeError(f"Boolean identifiers in {name}")
        if isinstance(value, int):
            number = value
        elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
            number = int(value)
        elif isinstance(value, str):
            number = int(value)
        else:
            raise TypeError(f"Invalid integer identifier in {name}")
        if not 0 <= number < 2**63:
            raise ValueError(f"Identifier outside int64 range in {name}")
        values.append(number)
    return np.asarray(values, dtype=np.int64)


def assert_equal(actual: float | FloatArray, expected: float | FloatArray, label: str) -> None:
    """Compare concrete real values using the documented tight tolerance"""
    if not np.allclose(actual, expected, rtol=RTOL, atol=ATOL):
        raise ValueError(f"{label} disagrees: actual={actual}, expected={expected}")


def normalize_table(df: pd.DataFrame) -> pd.DataFrame:
    """Validate physical columns and convert supported quantities to common units"""
    validate_column_labels(df)
    if df.empty:
        raise ValueError("The energetics table is empty")
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing energetics columns: {sorted(missing)}")
    result = df.copy()
    for name in REQUIRED:
        result[name] = integer_column(result, name) if name == "model_index" else numeric_column(result, name)
    for name in ("eps_b", "eps_th"):
        values = numeric_column(result, name)
        if np.any((values <= 0) | (values > 1)):
            raise ValueError(f"{name} must be in (0, 1]")
    for name, (key, target) in UNITS.items():
        converted = numeric_column(result, name) * unit_factor(df.attrs.get(key), target)
        if not np.all(np.isfinite(converted)):
            raise ValueError(f"Nonfinite converted values in {name}")
        result[name] = converted
        result.attrs[key] = target
    if np.any(numeric_column(result, "nu") <= 0):
        raise ValueError("Observation frequencies must be positive")
    return result


def result_metadata(df: pd.DataFrame) -> dict[str, float | str]:
    """Read normalized shared physical metadata"""
    for key in ("distance_value", "distance_unit", "burnt_mass_fraction"):
        if key not in df.attrs:
            raise ValueError(f"Missing physical metadata: {key}")
    distance = real_scalar(real_scalar(df.attrs["distance_value"]) * unit_factor(df.attrs["distance_unit"], "kpc"))
    fraction = real_scalar(df.attrs["burnt_mass_fraction"])
    if not 0 < fraction <= 1 or distance <= 0:
        raise ValueError("Distance and burnt_mass_fraction must be physically valid")
    return {"distance_value": distance, "distance_unit": "kpc", "burnt_mass_fraction": fraction}


def result_path(entry: RunEntry, relative_path: Path) -> Path:
    """Resolve an energetics path strictly within its RUN directory"""
    if relative_path.is_absolute():
        raise ValueError("energetics-path must be relative to the RUN directory")
    root = entry.input_path.parent.resolve()
    path = root / relative_path
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"energetics-path escapes its RUN directory: {relative_path}")
    return path


def combine_tables(entries: list[RunEntry], relative_path: Path, *, skip_missing: bool = False) -> pd.DataFrame:
    """Combine results only after validating inputs, units, parameters and provenance"""
    if not entries:
        raise ValueError("The RUN collection is empty")
    parameter = entries[0].parameter
    if parameter not in SWEEP_SPECS or any(entry.parameter != parameter for entry in entries):
        raise ValueError("All RUN entries must belong to one supported sweep parameter")
    spec = SWEEP_SPECS[parameter]
    if any(not 0 < real_scalar(entry.value) <= 1 for entry in entries):
        raise ValueError(f"Manifest {parameter} values must be in (0, 1]")
    tables: list[pd.DataFrame] = []
    family: object = None
    common: dict[str, float | str] | None = None
    schema: set[str] | None = None
    extra_units: dict[str, object] | None = None
    skipped = 0
    source_paths: list[str] = []
    for entry in entries:
        path = result_path(entry, relative_path)
        try:
            config = load_yaml(entry.input_path)
            current_family = sweep_family(config, parameter=parameter)
            if family is None:
                family = current_family
            elif current_family != family:
                raise ValueError(f"Input belongs to a different {parameter} sweep family")
            assert_equal(real_scalar(get_nested_value(config, spec.yaml_path)), entry.value, f"input {parameter}")
            fingerprint = calculate_config_hash(config)
            if entry.config_hash is not None and entry.config_hash != fingerprint:
                raise ValueError("Manifest hash disagrees with current input")
            if not path.exists():
                if skip_missing:
                    print(f"WARNING: {entry.run_name}: missing energetics file: {path}", flush=True)
                    skipped += 1
                    continue
                raise FileNotFoundError(f"Missing energetics file: {path}")
            raw: pd.DataFrame = FileReader.table_from_csv(path)
            validate_column_labels(raw)
            mapped_unit_keys = {key for key, _ in UNITS.values()} | {"distance_unit"}
            current_extra_units: dict[str, object] = {}
            for key, value in raw.attrs.items():
                if isinstance(key, str) and key.endswith("_unit") and key not in mapped_unit_keys:
                    current_extra_units[key] = value
            if extra_units is None:
                extra_units = current_extra_units
            elif current_extra_units != extra_units:
                raise ValueError("Additional unit metadata differs; an explicit conversion mapping is required")
            current_schema: set[str] = set()
            for column in raw.columns:
                if not isinstance(column, str):
                    raise TypeError("Result columns must have string names")
                if column not in IDENTIFIERS:
                    current_schema.add(column)
            if schema is None:
                schema = current_schema
            elif current_schema != schema:
                raise ValueError(f"Result schema differs: missing={sorted(schema-current_schema)}, extra={sorted(current_schema-schema)}")
            df = normalize_table(raw)
            metadata = result_metadata(df)
            assert_equal(numeric_column(df, parameter), entry.value, f"result {parameter}")
            for name in ("eps_b", "eps_th"):
                expected = real_scalar(get_nested_value(config, SWEEP_SPECS[name].yaml_path))
                if not 0 < expected <= 1:
                    raise ValueError(f"Input {name} must be in (0, 1]")
                assert_equal(numeric_column(df, name), expected, f"result {name}")
            common_path = "model_parameters.shock.common.molecular_weight"
            mu_e = real_scalar(get_nested_value(config, common_path + ".mu_e"))
            mu = real_scalar(get_nested_value(config, common_path + ".mu"))
            if mu_e <= 0 or mu <= 0:
                raise ValueError("Input molecular weights must be positive")
            assert_equal(numeric_column(df, "mu_e_mu"), mu_e / mu, "result mu_e_mu")
            expected_distance = real_scalar(
                real_scalar(get_nested_value(config, "model_parameters.distance.value"))
                * unit_factor(get_nested_value(config, "model_parameters.distance.unit"), "kpc")
            )
            assert_equal(real_scalar(metadata["distance_value"]), expected_distance, "result distance")
            if common is None:
                common = metadata
            else:
                for key in ("distance_value", "burnt_mass_fraction"):
                    assert_equal(real_scalar(metadata[key]), real_scalar(common[key]), key)
            if "run_id" in df:
                assert_equal(numeric_column(df, "run_id"), entry.run_id, "result run_id")
            validate_run_names(df, entry.run_name)
            if "result_hash_verified" in df:
                boolean_values(require_series(df, "result_hash_verified"), "Result result_hash_verified")
            hashes: list[object] = []
            if "config_hash" in raw.attrs:
                hashes.append(raw.attrs["config_hash"])
            if "config_hash" in raw:
                hashes.extend(require_series(raw, "config_hash").tolist())
            if any(value != fingerprint for value in hashes):
                raise ValueError("Result hash disagrees with current input")
            if not hashes:
                print(f"WARNING: {entry.run_name}: result config_hash unavailable; stale results cannot be excluded", flush=True)
            df["run_id"] = entry.run_id
            df["run_name"] = entry.run_name
            df["config_hash"] = fingerprint
            df["result_hash_verified"] = bool(hashes)
            if has_duplicate_results(df):
                raise ValueError("Duplicate result identities (run_id, nu, model_index)")
            tables.append(df)
            source_paths.append(str(path.resolve()))
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{entry.run_name}, input={entry.input_path}, result={path}: {error}") from error
    if not tables or common is None or extra_units is None:
        raise ValueError("No RUN contributed energetics results")
    # Assign attrs explicitly rather than relying on pandas' attrs propagation
    result = pd.concat([table.assign() for table in tables], ignore_index=True)
    if has_duplicate_results(result):
        raise ValueError("Duplicate result identities across RUN entries")
    result.attrs = {**common, **extra_units, **{key: target for key, target in UNITS.values()},
                    "included_runs": len(tables), "skipped_runs": skipped,
                    "sweep_parameter": parameter}
    verified = all(boolean_values(require_series(result, "result_hash_verified"), "result_hash_verified"))
    result.attrs["provenance_complete"] = "true" if verified else "false"
    print(f"Included RUNs: {len(tables)}; skipped RUNs: {skipped}", flush=True)
    # Keep source paths in memory for output collision checks, not CSV metadata
    result.attrs["_source_paths"] = source_paths
    return result


def write_output(df: pd.DataFrame, output: Path, protected: list[Path]) -> None:
    """Atomically publish a validated table without overwriting any source files"""
    sources = {path.resolve() for path in protected}
    raw_paths: object = df.attrs.get("_source_paths", [])
    if not isinstance(raw_paths, list):
        raise TypeError("Internal source paths must be a list")
    for path in raw_paths:
        if not isinstance(path, str):
            raise TypeError("Internal source paths must contain strings")
        sources.add(Path(path))
    if output.resolve() in sources:
        raise ValueError("Output must differ from the manifest, inputs and source result files")
    output.parent.mkdir(parents=True, exist_ok=True)
    clean = df.copy()
    clean.attrs.pop("_source_paths", None)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(dir=output.parent, prefix=".energetics_", suffix=".csv", delete=False) as file:
            temporary = Path(file.name)
        FileWriter.df_to_csv(path=temporary, df=clean)
        temporary.replace(output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-index", type=Path, required=True)
    parser.add_argument("--energetics-path", type=Path, default=Path("energetics.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()
    try:
        entries = read_manifest(args.sweep_index)
        df = combine_tables(entries, args.energetics_path, skip_missing=args.skip_missing)
        write_output(df, args.output, [
            args.sweep_index,
            args.sweep_index.with_suffix(".plan.json"),
            *(entry.input_path for entry in entries),
            *(result_path(entry, args.energetics_path) for entry in entries),
        ])
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
