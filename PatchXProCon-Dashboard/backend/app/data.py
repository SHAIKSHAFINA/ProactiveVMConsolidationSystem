"""
CSV reading utilities — all reads are read-only and path-validated.
"""
from __future__ import annotations

import csv
import io
import math
import re
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np


def read_csv_safe(path: Path) -> pd.DataFrame:
    """Read a CSV file; raise FileNotFoundError with a clear message if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")
    return pd.read_csv(path)


def summary_to_dict(path: Path) -> dict[str, Any]:
    """Parse a simulation_summary.csv (metric,value,notes) into a plain dict."""
    df = read_csv_safe(path)
    # Strip surrounding quotes that the Java writer adds
    df.columns = [c.strip('"') for c in df.columns]
    df["metric"] = df["metric"].str.strip('"')
    df["value"] = df["value"].str.strip('"')
    return dict(zip(df["metric"], df["value"]))


def policy_comparison_df(path: Path) -> pd.DataFrame:
    return read_csv_safe(path)


def _strip_csv_quotes(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [column.strip('"') for column in df.columns]

    def clean(value: Any) -> Any:
        if isinstance(value, str):
            return value.strip('"')
        return value if pd.isna(value) else str(value)

    for column in df.columns:
        df[column] = df[column].map(clean)
    return df


def migration_events_df(path: Path) -> pd.DataFrame:
    return _strip_csv_quotes(read_csv_safe(path))


def decision_plan_df(path: Path) -> pd.DataFrame:
    return _strip_csv_quotes(read_csv_safe(path))


def host_utilization_df(path: Path) -> pd.DataFrame:
    return _strip_csv_quotes(read_csv_safe(path))


def final_placement_df(path: Path) -> pd.DataFrame:
    return _strip_csv_quotes(read_csv_safe(path))


@lru_cache(maxsize=8)
def _input_csv_metadata_cached(
    path_string: str,
    modified_at: int,
    changed_at: int,
    size: int,
    device: int,
    inode: int,
) -> dict[str, Any]:
    path = Path(path_string)
    if not path.exists():
        return {"exists": False, "path": str(path)}
    df = pd.read_csv(path, nrows=5)
    columns = list(df.columns)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in columns]
    vm_ids: set[str] = set()
    missing_values: dict[str, int] = {}
    errors = [f"Missing required columns: {', '.join(missing_columns)}"] if missing_columns else []
    previous_timestamp: pd.Timestamp | None = None
    total_rows = 0
    if not missing_columns:
        inspect_columns = list(dict.fromkeys(REQUIRED_COLUMNS + [column for column in OPTIONAL_COLUMNS if column in columns]))
        for chunk in pd.read_csv(path, usecols=inspect_columns, chunksize=50000):
            total_rows += len(chunk)
            for column, count in chunk.isna().sum().items():
                if count:
                    missing_values[column] = missing_values.get(column, 0) + int(count)
            vm_ids.update(chunk["vm_id"].dropna().astype(str).unique())
            if chunk[REQUIRED_COLUMNS].isna().any().any():
                errors.append("One or more required columns contain missing values.")
            if chunk["vm_id"].astype(str).str.strip().eq("").any():
                errors.append("Column 'vm_id' contains empty values.")
            timestamps = pd.to_datetime(chunk["decision_time"], errors="coerce")
            if timestamps.isna().any():
                errors.append("Column 'decision_time' contains unparseable timestamps.")
            elif not timestamps.is_monotonic_increasing or (
                previous_timestamp is not None and timestamps.iloc[0] < previous_timestamp
            ):
                errors.append("Rows are not globally chronological by 'decision_time'.")
            if not timestamps.empty:
                previous_timestamp = timestamps.iloc[-1]
            for column in inspect_columns[2:]:
                if column not in chunk.columns:
                    continue
                values = pd.to_numeric(chunk[column], errors="coerce")
                if values.isna().any() or not values.map(math.isfinite).all():
                    errors.append(f"Column '{column}' contains missing, non-numeric, or non-finite values.")
                elif column in ("risk_probability", "risk_threshold"):
                    if ((values < 0) | (values > 1)).any():
                        errors.append(f"Column '{column}' values must be within [0, 1].")
                elif (values < 0).any():
                    errors.append(f"Column '{column}' must contain non-negative values.")
            if errors and missing_columns:
                break
    else:
        with path.open(encoding="utf-8", newline="") as stream:
            total_rows = max(sum(1 for _ in stream) - 1, 0)
    return {
        "exists": True,
        "columns": columns,
        "total_rows": total_rows,
        "vm_count": len(vm_ids),
        "sample_rows": df.head(5).to_dict(orient="records"),
        "validation": {
            "valid": bool(total_rows > 0 and not errors),
            "errors": list(dict.fromkeys(errors)),
            "warnings": [],
            "missing_values": missing_values,
        },
    }


def input_csv_metadata(path: Path) -> dict[str, Any]:
    """Return cached metadata for the configured prediction CSV."""
    if not path.exists():
        return {"exists": False}
    stat = path.stat()
    return _input_csv_metadata_cached(
        str(path.resolve()),
        stat.st_mtime_ns,
        stat.st_ctime_ns,
        stat.st_size,
        stat.st_dev,
        stat.st_ino,
    )


def _validate_raw_bitbrains_file_uncached(path: Path) -> dict[str, Any]:
    """Validate a combined raw Bitbrains trace without loading it all into memory."""
    errors: list[str] = []
    warnings: list[str] = []
    missing_values: dict[str, int] = {}
    vm_ids: set[str] = set()
    closed_vm_ids: set[str] = set()
    previous_vm: str | None = None
    columns: list[str] = []
    total_rows = 0
    negative_values: dict[str, int] = {}

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            header_line = stream.readline()
        delimiter = max(
            (",", ";"),
            key=lambda candidate: len(
                next(csv.reader([header_line], delimiter=candidate), [])
            ),
        )
        raw_columns = next(csv.reader([header_line], delimiter=delimiter), [])
        columns = [str(column).strip().rstrip(";").strip() for column in raw_columns]
        if len(columns) != len(set(columns)):
            errors.append("Raw Bitbrains CSV contains duplicate column names.")
        vm_columns = [column for column in ("VM_ID", "vm_id") if column in columns]
        vm_column = vm_columns[0] if len(vm_columns) == 1 else None
        required = ["Timestamp [ms]", *RAW_RESOURCE_COLUMNS]
        missing = [column for column in required if column not in columns]
        if len(vm_columns) > 1:
            errors.append("Raw Bitbrains CSV contains both 'VM_ID' and 'vm_id' columns.")
        elif vm_column is None:
            missing.append("VM_ID or vm_id")
        if missing:
            errors.append(
                "Raw Bitbrains CSV is missing required columns: " + ", ".join(missing) + "."
            )
        if errors:
            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "row_count": 0,
                "vm_count": 0,
                "columns": columns,
                "missing_values": missing_values,
                "preview": [],
                "dataset_kind": "raw_or_incompatible_csv",
            }

        selected_columns = [vm_column, *required]
        reader = pd.read_csv(
            path,
            sep=delimiter,
            usecols=selected_columns,
            chunksize=100_000,
            encoding="utf-8-sig",
        )
        preview: list[dict[str, Any]] = []
        for chunk in reader:
            chunk.columns = [str(column).strip().rstrip(";").strip() for column in chunk.columns]
            total_rows += len(chunk)
            if len(preview) < 5:
                preview.extend(chunk.head(5 - len(preview)).to_dict("records"))

            identifiers = chunk[vm_column]
            if identifiers.isna().any() or identifiers.astype(str).str.strip().eq("").any():
                errors.append(f"Column '{vm_column}' contains empty VM IDs.")
            normalized_ids = identifiers.astype(str).str.strip()
            vm_ids.update(normalized_ids[identifiers.notna()].unique())
            block_starts = normalized_ids.ne(normalized_ids.shift())
            for value in normalized_ids[block_starts]:
                if value != previous_vm:
                    if previous_vm is not None:
                        closed_vm_ids.add(previous_vm)
                    if value in closed_vm_ids:
                        errors.append(
                            "Rows for each VM must be contiguous in the combined raw CSV "
                            "for bounded-memory inference."
                        )
                        break
                    previous_vm = value

            timestamps = pd.to_numeric(chunk["Timestamp [ms]"], errors="coerce")
            invalid_timestamps = timestamps.isna() | ~np.isfinite(
                timestamps.to_numpy(dtype=np.float64, na_value=np.nan)
            )
            if invalid_timestamps.any():
                errors.append(
                    "Column 'Timestamp [ms]' must contain finite Unix-second timestamps "
                    "as used by the saved-model preprocessing."
                )
            elif pd.to_datetime(
                timestamps, unit="s", errors="coerce"
            ).isna().any():
                errors.append("Column 'Timestamp [ms]' contains out-of-range timestamps.")

            for column in RAW_RESOURCE_COLUMNS:
                source_values = chunk[column]
                numeric = pd.to_numeric(source_values, errors="coerce")
                if (source_values.notna() & numeric.isna()).any():
                    errors.append(f"Raw resource column '{column}' contains non-numeric values.")
                if np.isinf(
                    numeric.to_numpy(dtype=np.float64, na_value=np.nan)
                ).any():
                    errors.append(f"Raw resource column '{column}' contains non-finite values.")
                missing_count = int(numeric.isna().sum())
                if missing_count:
                    missing_values[column] = missing_values.get(column, 0) + missing_count
                negative_count = int((numeric < 0).sum())
                if negative_count:
                    negative_values[column] = negative_values.get(column, 0) + negative_count

        if total_rows == 0:
            errors.append("Raw Bitbrains CSV contains no data rows.")
        if negative_values:
            warnings.append(
                "Negative resource measurements are treated as missing by the existing "
                "training-time preprocessing: "
                + ", ".join(
                    f"{column}={count}" for column, count in sorted(negative_values.items())
                )
                + "."
            )
        if missing_values:
            warnings.append(
                "Missing resource values are preserved; any affected 5-minute model "
                "windows are skipped rather than imputed."
            )
        return {
            "valid": bool(total_rows > 0 and not errors),
            "errors": list(dict.fromkeys(errors)),
            "warnings": warnings,
            "row_count": total_rows,
            "vm_count": len(vm_ids),
            "vm_ids": sorted(vm_ids),
            "columns": columns,
            "missing_values": missing_values,
            "preview": preview,
            "dataset_kind": "raw_workload_csv",
        }
    except (
        OSError,
        UnicodeDecodeError,
        ValueError,
        pd.errors.ParserError,
    ) as exc:
        return {
            "valid": False,
            "errors": [f"Cannot validate raw Bitbrains CSV: {exc}"],
            "warnings": warnings,
            "row_count": total_rows,
            "vm_count": len(vm_ids),
            "vm_ids": sorted(vm_ids),
            "columns": columns,
            "missing_values": missing_values,
            "preview": [],
            "dataset_kind": "raw_or_incompatible_csv",
        }


@lru_cache(maxsize=8)
def _validate_raw_bitbrains_file_cached(
    path_string: str,
    modified_at: int,
    changed_at: int,
    size: int,
    device: int,
    inode: int,
) -> dict[str, Any]:
    path = Path(path_string)
    result = _validate_raw_bitbrains_file_uncached(path)
    final_stat = path.stat()
    if (
        final_stat.st_mtime_ns != modified_at
        or final_stat.st_ctime_ns != changed_at
        or final_stat.st_size != size
        or final_stat.st_dev != device
        or final_stat.st_ino != inode
    ):
        raise RuntimeError("The raw Bitbrains CSV changed during validation; retry when it is stable.")
    return result


def validate_raw_bitbrains_file(path: Path) -> dict[str, Any]:
    """Validate a stable raw file, reusing results only for an unchanged file identity."""
    resolved_path = path.resolve()
    stat = resolved_path.stat()
    return _validate_raw_bitbrains_file_cached(
        str(resolved_path),
        stat.st_mtime_ns,
        stat.st_ctime_ns,
        stat.st_size,
        stat.st_dev,
        stat.st_ino,
    )


REQUIRED_COLUMNS = [
    "vm_id",
    "decision_time",
    "risk_probability",
    "risk_threshold",
    "current_cpu_pct",
    "current_memory_kb",
    "forecast_cpu_max",
    "forecast_memory_max",
]

RAW_RESOURCE_COLUMNS = [
    "CPU usage [%]",
    "Memory usage [KB]",
    "Disk read throughput [KB/s]",
    "Disk write throughput [KB/s]",
    "Network received throughput [KB/s]",
    "Network transmitted throughput [KB/s]",
]

RAW_REQUIRED_COLUMNS = ["Timestamp [ms]", *RAW_RESOURCE_COLUMNS]

OPTIONAL_COLUMNS = [
    "current_disk_read_kbps",
    "current_disk_write_kbps",
    "forecast_disk_read_max",
    "forecast_disk_write_max",
]


def upload_compatibility_note(
    dataset_kind: str,
    columns: list[str],
    filename: str | None = None,
) -> str:
    if dataset_kind == "prediction_csv":
        return (
            "Prepared predictions are accepted as recorded forecast and risk inputs; "
            "they are not fresh model inference."
        )
    if dataset_kind == "raw_workload_csv":
        return (
            "The raw workload matches the verified Bitbrains training schema. "
            "Compatible preprocessing and fresh model inference are required before CloudSim."
        )
    if dataset_kind == "incompatible_raw_workload_csv":
        return (
            "Alibaba raw workload data is preserved without conversion. Its memory and "
            "disk-space percentages are not the memory-KB and disk-throughput inputs "
            "required by the trained model, so inference and CloudSim are disabled."
        )

    normalized_columns = {
        re.sub(r"[^a-z0-9]+", "_", column.lower()).strip("_")
        for column in columns
    }
    alibaba_markers = {
        "machine_id",
        "cpu_util_percent",
        "mem_util_percent",
        "time_stamp",
        "timestamp",
        "start_time",
        "task_index",
        "job_id",
        "instance_num",
    }
    is_alibaba_trace = (
        (filename is not None and "alibaba" in filename.lower())
        or (
            "machine_id" in normalized_columns
            and len(normalized_columns & alibaba_markers) >= 3
        )
    )
    if is_alibaba_trace:
        return (
            "This looks like Alibaba Cluster Trace data, whose schema differs from the "
            "verified Bitbrains training input. It needs an explicit, trace-specific mapping "
            "and compatible preprocessing/inference before simulation. Missing features, "
            "forecasts, disk measurements, and risk probabilities will not be fabricated."
        )
    return (
        "This CSV does not match the prepared-prediction schema or verified Bitbrains raw "
        "workload schema. It needs compatible preprocessing and inference before simulation. "
        "Missing features, forecasts, disk measurements, and risk probabilities will not "
        "be fabricated."
    )


def simulator_compatibility_errors(path: Path) -> list[str]:
    """Validate the selected CSV against the frozen simulator's Java CSV reader."""
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            columns = next(csv.reader(stream), [])
    except (OSError, ValueError, UnicodeDecodeError, csv.Error) as exc:
        return [f"Cannot read simulator input CSV: {type(exc).__name__}."]

    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        return [f"Required simulator columns are missing: {', '.join(missing)}."]
    if len(columns) != len(set(columns)):
        return ["Duplicate CSV column names are not supported by the simulator."]

    inspect_columns = list(dict.fromkeys(
        REQUIRED_COLUMNS + [column for column in OPTIONAL_COLUMNS if column in columns]
    ))
    errors: list[str] = []
    previous_timestamp: pd.Timestamp | None = None
    row_number = 1
    try:
        for chunk in pd.read_csv(
            path,
            usecols=inspect_columns,
            dtype={"vm_id": str, "decision_time": str},
            chunksize=50_000,
        ):
            required_nulls = chunk[REQUIRED_COLUMNS].isna()
            if required_nulls.any().any():
                bad_column = str(required_nulls.columns[required_nulls.any()][0])
                errors.append(f"CSV row {row_number + 1}: required field '{bad_column}' is empty.")
                break

            empty_vm = chunk["vm_id"].str.strip().eq("")
            if empty_vm.any():
                first_empty = int(empty_vm.to_numpy().argmax())
                errors.append(f"CSV row {row_number + first_empty + 1}: vm_id is empty.")
                break

            timestamps = chunk["decision_time"]
            for offset, value in enumerate(timestamps):
                try:
                    parsed = datetime.fromisoformat(value)
                except (TypeError, ValueError):
                    errors.append(
                        f"CSV row {row_number + offset + 1}: decision_time '{value}' "
                        "is not supported; use local 'YYYY-MM-DD HH:MM:SS' or ISO local datetime."
                    )
                    break
                if parsed.tzinfo is not None or not re.search(r"[T ]\d{2}:\d{2}", value):
                    errors.append(
                        f"CSV row {row_number + offset + 1}: decision_time must be a timezone-free "
                        "local datetime (Java LocalDateTime format)."
                    )
                    break
            if errors:
                break

            parsed_times = pd.to_datetime(timestamps, errors="coerce")
            if parsed_times.isna().any():
                errors.append(f"CSV row {row_number + 1}: decision_time contains an invalid timestamp.")
                break
            if not parsed_times.is_monotonic_increasing or (
                previous_timestamp is not None and parsed_times.iloc[0] < previous_timestamp
            ):
                errors.append("Rows are not globally chronological by 'decision_time'.")
                break
            if not parsed_times.empty:
                previous_timestamp = parsed_times.iloc[-1]

            for column in inspect_columns[2:]:
                values = pd.to_numeric(chunk[column], errors="coerce")
                if values.isna().any() or not values.map(math.isfinite).all():
                    errors.append(
                        f"CSV row {row_number + 1}: '{column}' must contain finite numeric values."
                    )
                    break
                if column in ("risk_probability", "risk_threshold"):
                    if ((values < 0) | (values > 1)).any():
                        errors.append(f"'{column}' must be within [0, 1].")
                        break
                elif (values < 0).any():
                    errors.append(f"'{column}' must be non-negative.")
                    break
            if errors:
                break

            if chunk.select_dtypes(include=["object"]).apply(
                lambda column: column.astype(str).str.contains(r"[\r\n]", regex=True).any()
            ).any():
                errors.append(
                    f"CSV rows near {row_number + 1} contain embedded line breaks, "
                    "which the simulator's line-based reader cannot process."
                )
                break
            row_number += len(chunk)
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        errors.append(f"Cannot parse simulator input CSV: {type(exc).__name__}.")
    return errors


def validate_upload_csv(content: bytes, filename: str | None = None) -> dict[str, Any]:
    """
    Validate either a prepared prediction CSV or a Bitbrains raw workload CSV.
    """
    errors: list[str] = []
    warnings: list[str] = []
    duplicate_headers = False
    delimiter = ","
    try:
        first_line = content.decode("utf-8-sig").splitlines()[0]
        header_options = {
            candidate: next(csv.reader([first_line], delimiter=candidate), [])
            for candidate in (",", ";")
        }
        delimiter, raw_header = max(
            header_options.items(), key=lambda item: len(item[1])
        )
        duplicate_headers = len(raw_header) != len(set(raw_header))
    except (UnicodeDecodeError, csv.Error, IndexError):
        raw_header = []

    raw_schema_present = set(RAW_REQUIRED_COLUMNS).issubset(raw_header)
    prediction_schema_present = set(REQUIRED_COLUMNS).issubset(raw_header)
    try:
        df = pd.read_csv(io.BytesIO(content), sep=delimiter, encoding="utf-8-sig")
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError, OSError) as exc:
        return {
            "valid": False,
            "errors": [f"Cannot parse CSV: {exc}"],
            "simulator_compatible": False,
            "simulator_errors": [f"Cannot parse CSV: {exc}"],
            "warnings": [],
            "row_count": 0,
            "vm_count": 0,
            "columns": [],
            "missing_values": {},
            "dataset_kind": "raw_or_incompatible_csv",
        }

    cols = list(df.columns)
    missing_values_all = {
        str(column): int(count)
        for column, count in df.isna().sum().items()
        if count
    }
    normalized_columns = {
        re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_"): column
        for column in cols
    }
    alibaba_candidate = (
        (filename is not None and "alibaba" in filename.lower())
        or (
            "machine_id" in normalized_columns
            and {"cpu_util_percent", "mem_util_percent"}.issubset(normalized_columns)
        )
    )
    if alibaba_candidate:
        if duplicate_headers or len(cols) != len(set(cols)):
            errors.append("CSV contains duplicate column names.")
        if not cols or df.empty:
            errors.append("CSV must contain a header and at least one data row.")
        machine_column = normalized_columns.get("machine_id")
        timestamp_column = next(
            (
                normalized_columns[name]
                for name in ("time_stamp", "timestamp", "start_time", "time")
                if name in normalized_columns
            ),
            None,
        )
        required_alibaba = ("cpu_util_percent", "mem_util_percent")
        missing_alibaba = [
            name for name in required_alibaba if name not in normalized_columns
        ]
        if machine_column is None:
            errors.append("Alibaba raw data must identify each machine with 'machine_id'.")
        if timestamp_column is None:
            errors.append(
                "Alibaba raw data must include a recognizable timestamp column "
                "('timestamp', 'time_stamp', 'start_time', or 'time')."
            )
        if missing_alibaba:
            errors.append(
                "Alibaba raw data is missing workload columns: "
                + ", ".join(missing_alibaba)
                + "."
            )
        vm_count = 0
        if machine_column is not None:
            machine_ids = df[machine_column]
            if machine_ids.isna().any() or machine_ids.astype(str).str.strip().eq("").any():
                errors.append(f"Column '{machine_column}' contains empty values.")
            vm_count = int(machine_ids.nunique())
        if timestamp_column is not None:
            timestamps = pd.to_datetime(df[timestamp_column], errors="coerce")
            if timestamps.isna().any():
                errors.append(
                    f"Column '{timestamp_column}' contains unparseable timestamps."
                )
        numeric_names = [*required_alibaba]
        if "disk_util_percent" in normalized_columns:
            numeric_names.append("disk_util_percent")
        for name in numeric_names:
            column = normalized_columns[name]
            values = pd.to_numeric(df[column], errors="coerce")
            invalid = df[column].notna() & values.isna()
            if invalid.any() or not values.dropna().map(math.isfinite).all():
                errors.append(f"Alibaba workload column '{column}' contains invalid numbers.")
            if values.isna().any():
                warnings.append(
                    f"Column '{column}' contains missing values; they are preserved, "
                    "not imputed."
                )
        if "disk_util_percent" in normalized_columns:
            warnings.append(
                "Disk-space percentage is capacity utilization, not disk read/write "
                "throughput; it cannot be used as a model input."
            )
        warnings.append(
            "Memory percentage is not memory usage in KB; trained-model inference is "
            "unavailable for this trace without a verified compatible preprocessing pipeline."
        )
        simulator_errors = [
            "Alibaba raw rows do not contain the four verified Bitbrains model channels "
            "and cannot be passed to CloudSim as prepared predictions."
        ]
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": list(dict.fromkeys(warnings)),
            "simulator_compatible": False,
            "simulator_errors": simulator_errors,
            "row_count": len(df),
            "vm_count": vm_count,
            "columns": cols,
            "optional_columns_present": [],
            "missing_values": missing_values_all,
            "preview": (
                df.head(5)
                .astype(object)
                .where(pd.notna(df.head(5)), None)
                .to_dict(orient="records")
            ),
            "dataset_kind": "incompatible_raw_workload_csv",
        }

    raw_candidate = (
        "Timestamp [ms]" in raw_header
        or any(column in raw_header for column in RAW_RESOURCE_COLUMNS)
    )
    if raw_candidate and not prediction_schema_present:
        missing_raw = [column for column in RAW_REQUIRED_COLUMNS if column not in cols]
        if duplicate_headers:
            errors.append("CSV contains duplicate column names.")
        if missing_raw:
            errors.append(
                "Raw Bitbrains CSV is missing training-schema columns: "
                + ", ".join(missing_raw)
                + "."
            )
        if not cols or df.empty:
            errors.append("CSV must contain a header and at least one data row.")
        if not missing_raw and not df.empty:
            timestamps = pd.to_numeric(df["Timestamp [ms]"], errors="coerce")
            if timestamps.isna().any() or not timestamps.map(math.isfinite).all():
                errors.append(
                    "Column 'Timestamp [ms]' must contain finite Unix-second timestamps "
                    "as used by the training preprocessing."
                )
            for column in RAW_RESOURCE_COLUMNS:
                numeric = pd.to_numeric(df[column], errors="coerce")
                invalid = df[column].notna() & numeric.isna()
                if invalid.any():
                    errors.append(
                        f"Raw resource column '{column}' contains non-numeric values."
                    )
                if numeric.isna().any():
                    warnings.append(
                        f"Raw resource column '{column}' contains missing values; "
                        "affected model windows will be skipped."
                    )
                if (numeric < 0).any():
                    warnings.append(
                        f"Negative values in '{column}' are treated as missing, matching "
                        "the training preprocessing."
                    )
            raw_vm_column = next(
                (column for column in ("VM_ID", "vm_id") if column in cols),
                None,
            )
            if raw_vm_column is not None:
                if (
                    df[raw_vm_column].isna().any()
                    or df[raw_vm_column].astype(str).str.strip().eq("").any()
                ):
                    errors.append(f"Column '{raw_vm_column}' contains empty VM IDs.")
                raw_vm_count = int(df[raw_vm_column].nunique())
            else:
                raw_vm_count = 1
        else:
            raw_vm_column = next(
                (column for column in ("VM_ID", "vm_id") if column in cols),
                None,
            )
            raw_vm_count = int(df[raw_vm_column].nunique()) if raw_vm_column else 0

        simulator_errors = [
            "Raw workload rows require PatchTST and Logistic Regression inference "
            "before they can be passed to CloudSim."
        ]
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": list(dict.fromkeys(warnings)),
            "simulator_compatible": False,
            "simulator_errors": simulator_errors,
            "row_count": len(df),
            "vm_count": raw_vm_count,
            "vm_ids": (
                sorted(df[raw_vm_column].dropna().astype(str).str.strip().unique().tolist())
                if raw_vm_column
                else ["1"]
            ),
            "columns": cols,
            "optional_columns_present": [],
            "missing_values": missing_values_all,
            "preview": (
                df.head(5)
                .astype(object)
                .where(pd.notna(df.head(5)), None)
                .to_dict(orient="records")
            ),
            "dataset_kind": (
                "raw_workload_csv" if raw_schema_present else "raw_or_incompatible_csv"
            ),
        }

    if not cols or df.empty:
        errors.append("CSV must contain a header and at least one data row.")
    if duplicate_headers or len(cols) != len(set(cols)):
        errors.append("CSV contains duplicate column names.")

    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")

    optional_present = [c for c in OPTIONAL_COLUMNS if c in cols]
    optional_missing = [c for c in OPTIONAL_COLUMNS if c not in cols]
    if optional_missing:
        warnings.append(f"Optional disk I/O columns absent: {', '.join(optional_missing)}")

    if not errors:
        required_missing = df[REQUIRED_COLUMNS].isna().sum()
        missing_values = {
            str(column): int(count)
            for column, count in required_missing.items()
            if count
        }
        if missing_values:
            errors.append(f"Required columns contain missing values: {missing_values}")
        else:
            missing_values = {}

        if df["vm_id"].astype(str).str.strip().eq("").any():
            errors.append("Column 'vm_id' contains empty values.")

        numeric_cols = [
            "current_cpu_pct",
            "current_memory_kb",
            "forecast_cpu_max",
            "forecast_memory_max",
            "risk_probability",
            "risk_threshold",
            *optional_present,
        ]
        for col in numeric_cols:
            try:
                values = pd.to_numeric(df[col], errors="raise")
            except (TypeError, ValueError):
                errors.append(f"Column '{col}' contains non-numeric values.")
                continue
            if not values.map(math.isfinite).all():
                errors.append(f"Column '{col}' contains non-finite values.")
            if col not in ("risk_probability", "risk_threshold") and (values < 0).any():
                errors.append(f"Column '{col}' must contain non-negative values.")
            if col in ("risk_probability", "risk_threshold") and (
                (values < 0) | (values > 1)
            ).any():
                errors.append(f"Column '{col}' values must be within [0, 1].")

        try:
            timestamps = pd.to_datetime(df["decision_time"], errors="coerce")
            if timestamps.isna().any():
                errors.append("Column 'decision_time' contains unparseable timestamps.")
            elif not timestamps.is_monotonic_increasing:
                errors.append("Rows must be globally chronological by 'decision_time'.")
        except (TypeError, ValueError) as exc:
            errors.append(f"Timestamp validation failed: {exc}")

        if missing_values_all and not missing_values:
            warnings.append(f"Optional columns contain missing values: {missing_values_all}")

    simulator_errors: list[str] = list(errors)
    if content.startswith(b"\xef\xbb\xbf"):
        simulator_errors.append(
            "UTF-8 byte-order mark is not supported in the simulator CSV header; remove the BOM."
        )
    if missing:
        simulator_errors.append(
            f"Required simulator columns are missing: {', '.join(missing)}."
        )
    elif "decision_time" in df.columns:
        for index, value in enumerate(df["decision_time"].astype(str), start=2):
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError:
                simulator_errors.append(
                    f"CSV row {index}: decision_time '{value}' is not supported; use "
                    "local 'YYYY-MM-DD HH:MM:SS' or ISO local datetime."
                )
                break
            if parsed.tzinfo is not None or not re.search(r"[T ]\d{2}:\d{2}", value):
                simulator_errors.append(
                    f"CSV row {index}: decision_time must be a timezone-free local datetime "
                    "(Java LocalDateTime format)."
                )
                break
        for column in df.select_dtypes(include=["object"]).columns:
            if df[column].astype(str).str.contains(r"[\r\n]", regex=True).any():
                simulator_errors.append(
                    f"Column '{column}' contains embedded line breaks, which the simulator's "
                    "line-based reader cannot process."
                )
                break

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "simulator_compatible": len(simulator_errors) == 0,
        "simulator_errors": simulator_errors,
        "row_count": len(df),
        "vm_count": int(df["vm_id"].nunique()) if "vm_id" in df.columns else 0,
        "vm_ids": (
            sorted(df["vm_id"].dropna().astype(str).str.strip().unique().tolist())
            if "vm_id" in df.columns
            else []
        ),
        "columns": cols,
        "optional_columns_present": optional_present,
        "missing_values": missing_values_all,
        "preview": (
            df.head(5)
            .astype(object)
            .where(pd.notna(df.head(5)), None)
            .to_dict(orient="records")
        ),
        "dataset_kind": (
            "prediction_csv"
            if prediction_schema_present
            else "raw_or_incompatible_csv"
        ),
    }
