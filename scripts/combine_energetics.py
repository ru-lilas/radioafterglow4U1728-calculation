"""Combine validated energetics tables from an eps_B sweep"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd

from module.utils import FileReader, FileWriter

if __package__:
    from .build_input import EPS_B_PATH, calculate_config_hash, get_nested_value, load_yaml, sweep_family
else:
    from build_input import EPS_B_PATH, calculate_config_hash, get_nested_value, load_yaml, sweep_family

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
    eps_b: float
    run_id: int
    run_name: str
    input_path: Path
    config_hash: str | None


def real_scalar(value: object) -> float:
    """Validate a finite real scalar before conversion"""
    if isinstance(value, np.generic):
        return real_scalar(value.item())
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
        required = {"eps_b", "run_id", "run_name", "input_path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing manifest columns: {sorted(required - set(reader.fieldnames or []))}")
        entries = []
        ids: set[int] = set()
        paths: set[Path] = set()
        values: set[float] = set()
        names: set[str] = set()
        for row in reader:
            eps_b = real_scalar(float(row["eps_b"]))
            run_id = int(row["run_id"])
            input_path = Path(row["input_path"])
            if not 0 < eps_b <= 1 or run_id < 1 or not row["run_name"] or not row["input_path"]:
                raise ValueError(f"Invalid manifest entry: {row}")
            if run_id in ids or input_path.resolve() in paths or eps_b in values or row["run_name"] in names:
                raise ValueError(f"Duplicate or ambiguous manifest entry: {row}")
            ids.add(run_id); paths.add(input_path.resolve()); values.add(eps_b); names.add(row["run_name"])
            entries.append(RunEntry(eps_b, run_id, row["run_name"], input_path, row.get("config_hash") or None))
    if not entries:
        raise ValueError("The sweep manifest is empty")
    return entries


def numeric_column(df: pd.DataFrame, name: str) -> np.ndarray:
    """Read a finite real numeric column without accepting boolean or complex values"""
    if df[name].map(lambda value: isinstance(value, (bool, complex, np.bool_, np.complexfloating))).any():
        raise TypeError(f"Invalid numeric values in {name}")
    values = pd.to_numeric(df[name], errors="raise").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Nonfinite values in {name}")
    return values


def assert_equal(actual: object, expected: object, label: str) -> None:
    """Compare real values using the documented tight floating-point tolerance"""
    if not np.allclose(actual, expected, rtol=RTOL, atol=ATOL):
        raise ValueError(f"{label} disagrees: actual={actual}, expected={expected}")


def normalize_table(df: pd.DataFrame) -> pd.DataFrame:
    """Validate columns and convert supported quantities to common output units"""
    if df.empty:
        raise ValueError("The energetics table is empty")
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing energetics columns: {sorted(missing)}")
    result = df.copy()
    for name in REQUIRED:
        values = numeric_column(result, name)
        if name == "model_index":
            identifiers = pd.to_numeric(result[name], errors="raise").to_numpy()
            if not np.all(identifiers == np.floor(identifiers)) or np.any(identifiers < 0) or np.any(identifiers >= 2**63):
                raise ValueError("model_index must contain nonnegative integer identifiers")
            result[name] = identifiers.astype(np.int64)
        else:
            result[name] = values
    for name, (key, target) in UNITS.items():
        source = df.attrs.get(key)
        if not isinstance(source, str) or not source:
            raise ValueError(f"Missing or invalid unit metadata: {key}")
        result[name] = u.Quantity(result[name].to_numpy(), source).to_value(target)
        if not np.all(np.isfinite(result[name])):
            raise ValueError(f"Nonfinite converted values in {name}")
        result.attrs[key] = target
    return result


def result_metadata(df: pd.DataFrame) -> dict[str, Any]:
    """Read normalized shared physical metadata"""
    for key in ("distance_value", "distance_unit", "burnt_mass_fraction"):
        if key not in df.attrs:
            raise ValueError(f"Missing physical metadata: {key}")
    distance = u.Quantity(real_scalar(df.attrs["distance_value"]), df.attrs["distance_unit"]).to_value("kpc")
    fraction = real_scalar(df.attrs["burnt_mass_fraction"])
    if not 0 < fraction <= 1 or distance <= 0:
        raise ValueError("Distance and burnt_mass_fraction must be physically valid")
    return {"distance_value": real_scalar(distance), "distance_unit": "kpc", "burnt_mass_fraction": fraction}


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
    tables = []
    family = None
    common = None
    schema = None
    extra_units = None
    skipped = 0
    source_paths: list[str] = []
    for entry in entries:
        path = result_path(entry, relative_path)
        try:
            config = load_yaml(entry.input_path)
            current_family = sweep_family(config)
            if family is None:
                family = current_family
            elif current_family != family:
                raise ValueError("Input belongs to a different eps_B sweep family")
            assert_equal(real_scalar(get_nested_value(config, EPS_B_PATH)), entry.eps_b, "input eps_b")
            fingerprint = calculate_config_hash(config)
            if entry.config_hash is not None and entry.config_hash != fingerprint:
                raise ValueError("Manifest hash disagrees with current input")
            if not path.exists():
                if skip_missing:
                    print(f"WARNING: {entry.run_name}: missing energetics file: {path}", flush=True)
                    skipped += 1
                    continue
                raise FileNotFoundError(f"Missing energetics file: {path}")
            raw = FileReader.table_from_csv(path)
            mapped_unit_keys = {key for key, _ in UNITS.values()} | {"distance_unit"}
            current_extra_units = {
                key: value for key, value in raw.attrs.items()
                if isinstance(key, str) and key.endswith("_unit") and key not in mapped_unit_keys
            }
            if extra_units is None:
                extra_units = current_extra_units
            elif current_extra_units != extra_units:
                raise ValueError("Additional unit metadata differs; an explicit conversion mapping is required")
            current_schema = set(raw.columns) - IDENTIFIERS
            if schema is None:
                schema = current_schema
            elif current_schema != schema:
                raise ValueError(f"Result schema differs: missing={sorted(schema-current_schema)}, extra={sorted(current_schema-schema)}")
            df = normalize_table(raw)
            metadata = result_metadata(df)
            assert_equal(df["eps_b"].to_numpy(), entry.eps_b, "result eps_b")
            shock = get_nested_value(config, "model_parameters.shock.common")
            eps_th = real_scalar(shock["eps_th"])
            mu_ratio = real_scalar(shock["molecular_weight"]["mu_e"]) / real_scalar(shock["molecular_weight"]["mu"])
            assert_equal(df["eps_th"].to_numpy(), eps_th, "result eps_th")
            assert_equal(df["mu_e_mu"].to_numpy(), mu_ratio, "result mu_e_mu")
            distance = get_nested_value(config, "model_parameters.distance")
            expected_distance = u.Quantity(real_scalar(distance["value"]), distance["unit"]).to_value("kpc")
            assert_equal(metadata["distance_value"], expected_distance, "result distance")
            if common is None:
                common = metadata
            else:
                for key in ("distance_value", "burnt_mass_fraction"):
                    assert_equal(metadata[key], common[key], key)
            if "run_id" in df:
                assert_equal(numeric_column(df, "run_id"), entry.run_id, "result run_id")
            if "run_name" in df and not df["run_name"].eq(entry.run_name).all():
                raise ValueError("Result run_name disagrees with manifest")
            hashes: list[object] = []
            if "config_hash" in raw.attrs:
                hashes.append(raw.attrs["config_hash"])
            if "config_hash" in raw:
                hashes.extend(raw["config_hash"].tolist())
            if any(value != fingerprint for value in hashes):
                raise ValueError("Result hash disagrees with current input")
            if not hashes:
                print(f"WARNING: {entry.run_name}: result config_hash unavailable; stale results cannot be excluded", flush=True)
            df["run_id"] = entry.run_id
            df["run_name"] = entry.run_name
            df["config_hash"] = fingerprint
            df["result_hash_verified"] = bool(hashes)
            if df.duplicated(["run_id", "nu", "model_index"]).any():
                raise ValueError("Duplicate result identities (run_id, nu, model_index)")
            tables.append(df)
            source_paths.append(str(path.resolve()))
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{entry.run_name}, input={entry.input_path}, result={path}: {error}") from error
    if not tables:
        raise ValueError("No RUN contributed energetics results")
    # Assign attrs explicitly rather than relying on pandas' attrs propagation
    result = pd.concat([table.assign() for table in tables], ignore_index=True)
    result.attrs = {**common, **extra_units, **{key: target for key, target in UNITS.values()},
                    "included_runs": len(tables), "skipped_runs": skipped}
    result.attrs["provenance_complete"] = "true" if result["result_hash_verified"].all() else "false"
    print(f"Included RUNs: {len(tables)}; skipped RUNs: {skipped}", flush=True)
    # Keep source paths in memory for output collision checks, not CSV metadata
    result.attrs["_source_paths"] = source_paths
    return result


def write_output(df: pd.DataFrame, output: Path, protected: list[Path]) -> None:
    """Atomically publish a validated table without overwriting any source files"""
    sources = {path.resolve() for path in protected}
    sources.update(Path(path) for path in df.attrs.get("_source_paths", []))
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
