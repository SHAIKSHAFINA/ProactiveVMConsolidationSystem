"""Dataset catalog, validation, upload, and active-dataset selection."""
from __future__ import annotations

import json
import hashlib
import csv
import sqlite3
import tempfile
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator, model_validator
from starlette.concurrency import run_in_threadpool

from ..config import settings
from ..data import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    input_csv_metadata,
    upload_compatibility_note,
    validate_upload_csv,
    validate_raw_bitbrains_file,
)
from ..inference import (
    OUTPUT_COLUMNS,
    generate_prediction_csv,
    model_inference_status,
)

router = APIRouter()
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
DEMO_DATASET_ID = "bitbrains"
ACTIVE_DATASET_FILE = settings.dashboard_uploads_dir / "active_dataset.json"
PREDICTION_COLUMNS = (
    "vm_id",
    "decision_time",
    "current_cpu_pct",
    "current_memory_kb",
    "forecast_cpu_max",
    "forecast_memory_max",
    "risk_probability",
    "risk_threshold",
    "risk_alert",
    *OPTIONAL_COLUMNS,
)
_inference_lock = threading.Lock()
_LOCAL_FILE_IDENTITY_FIELDS = {
    "local_changed_at_ns": "st_ctime_ns",
    "local_device": "st_dev",
    "local_inode": "st_ino",
}


def _local_file_identity(stat: Any) -> dict[str, int]:
    return {
        key: int(getattr(stat, attribute))
        for key, attribute in _LOCAL_FILE_IDENTITY_FIELDS.items()
    }


def _local_file_matches_metadata(metadata: dict[str, Any], stat: Any) -> bool:
    if (
        metadata.get("local_size_bytes") != stat.st_size
        or metadata.get("local_modified_at_ns") != stat.st_mtime_ns
    ):
        return False
    return all(
        metadata.get(key) == value
        for key, value in _local_file_identity(stat).items()
        if key in metadata
    )


class DatasetSelection(BaseModel):
    dataset_id: str
    vm_ids: list[str] | None = Field(default=None, max_length=50)
    start_time: str | None = None
    end_time: str | None = None

    @field_validator("vm_ids")
    @classmethod
    def validate_vm_ids(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("vm_ids must not contain empty values.")
        if len(normalized) != len(set(normalized)):
            raise ValueError("vm_ids must not contain duplicates.")
        return normalized

    @field_validator("start_time", "end_time")
    @classmethod
    def validate_selection_datetime(cls, value: str | None, info: Any) -> str | None:
        if value is None:
            return None
        try:
            parsed = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{info.field_name} must be an ISO local datetime.") from exc
        if parsed.tzinfo is not None:
            raise ValueError(f"{info.field_name} must not include a timezone.")
        return value

    @model_validator(mode="after")
    def validate_time_window(self) -> "DatasetSelection":
        if self.start_time and self.end_time:
            if pd.Timestamp(self.start_time) > pd.Timestamp(self.end_time):
                raise ValueError("start_time must be earlier than or equal to end_time.")
        return self


def _merge_inference_batch(
    existing_path: Path,
    batch_path: Path,
    output_path: Path,
) -> tuple[int, list[str]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    database_file = tempfile.NamedTemporaryFile(
        prefix="patchx-inference-merge-",
        suffix=".sqlite3",
        dir=output_path.parent,
        delete=False,
    )
    database_path = Path(database_file.name)
    database_file.close()
    connection = sqlite3.connect(database_path)
    column_sql = ", ".join(f'"{column}" TEXT' for column in OUTPUT_COLUMNS)
    connection.execute(
        f"CREATE TABLE predictions ({column_sql}, PRIMARY KEY (vm_id, decision_time))"
    )
    insert_sql = (
        "INSERT OR REPLACE INTO predictions ("
        + ", ".join(f'"{column}"' for column in OUTPUT_COLUMNS)
        + ") VALUES ("
        + ", ".join("?" for _ in OUTPUT_COLUMNS)
        + ")"
    )

    def import_csv(path: Path) -> None:
        if not path.is_file():
            return
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not set(OUTPUT_COLUMNS).issubset(reader.fieldnames):
                raise ValueError(f"Inference batch '{path.name}' has an invalid schema.")
            batch: list[tuple[str, ...]] = []
            for record in reader:
                batch.append(tuple(record[column] for column in OUTPUT_COLUMNS))
                if len(batch) == 10_000:
                    connection.executemany(insert_sql, batch)
                    batch.clear()
            if batch:
                connection.executemany(insert_sql, batch)
        connection.commit()

    temporary_output = output_path.with_name(f"{output_path.stem}-{uuid.uuid4().hex}.tmp.csv")
    try:
        import_csv(existing_path)
        import_csv(batch_path)
        total_rows = int(
            connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        )
        vm_ids = [
            str(row[0])
            for row in connection.execute(
                'SELECT DISTINCT "vm_id" FROM predictions ORDER BY "vm_id"'
            )
        ]
        with temporary_output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(OUTPUT_COLUMNS)
            cursor = connection.execute(
                'SELECT * FROM predictions ORDER BY "decision_time", "vm_id"'
            )
            while rows := cursor.fetchmany(10_000):
                writer.writerows(rows)
        temporary_output.replace(output_path)
        return total_rows, vm_ids
    finally:
        connection.close()
        temporary_output.unlink(missing_ok=True)
        database_path.unlink(missing_ok=True)


def inference_checkpoint_path(
    dataset_id: str,
    start_time: str | None,
    end_time: str | None,
) -> Path:
    window_key = hashlib.sha256(
        f"{start_time or ''}\0{end_time or ''}".encode("utf-8")
    ).hexdigest()[:16]
    return (
        settings.dashboard_runs_dir
        / "inference"
        / dataset_id
        / f"predictions-{window_key}.sqlite3"
    )


def infer_and_merge_dataset(
    input_path: Path,
    output_path: Path,
    temporary_path: Path,
    source_filename: str,
    vm_ids: list[str] | None,
    start_time: str | None,
    end_time: str | None,
    on_stage: Callable[[str], None] | None = None,
    on_vm_complete: Callable[[int, str | None], None] | None = None,
) -> dict[str, Any]:
    with _inference_lock:
        try:
            result = generate_prediction_csv(
                input_path,
                temporary_path,
                source_filename,
                on_stage=on_stage,
                vm_ids=set(vm_ids) if vm_ids else None,
                start_time=start_time,
                end_time=end_time,
                checkpoint_path=inference_checkpoint_path(
                    output_path.parent.name,
                    start_time,
                    end_time,
                ),
                on_vm_complete=on_vm_complete,
            )
            if result["total_row_count"] == 0:
                raise ValueError(
                    "No continuous 108-sample, five-minute workload segment was found; "
                    "no forecasts or risk scores were generated."
                )
            total_rows, completed_vm_ids = _merge_inference_batch(
                output_path,
                temporary_path,
                output_path,
            )
            return {
                **result,
                "generated_row_count": result["row_count"],
                "checkpoint_row_count": result["total_row_count"],
                "checkpoint_vm_count": result["total_vm_count"],
                "total_row_count": total_rows,
                "completed_vm_ids": completed_vm_ids,
                "completed_vm_count": len(completed_vm_ids),
            }
        finally:
            temporary_path.unlink(missing_ok=True)


def _active_id() -> str:
    if not ACTIVE_DATASET_FILE.exists():
        return DEMO_DATASET_ID
    try:
        data = json.loads(ACTIVE_DATASET_FILE.read_text(encoding="utf-8"))
        return str(data["dataset_id"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Unable to read active dataset state: {exc}") from exc


def get_active_dataset_id() -> str:
    return _active_id()


def _set_active_id(dataset_id: str) -> None:
    temporary_path = ACTIVE_DATASET_FILE.with_name(
        f".{ACTIVE_DATASET_FILE.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary_path.write_text(
            json.dumps({"dataset_id": dataset_id}, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(ACTIVE_DATASET_FILE)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise


def _duplicate_dataset_id(content_sha256: str) -> str | None:
    for metadata_path in settings.dashboard_uploads_dir.glob("*.json"):
        if metadata_path.name == ACTIVE_DATASET_FILE.name:
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Unable to inspect dataset metadata '{metadata_path.stem}': {exc}"
            ) from exc
        dataset_id = metadata.get("dataset_id")
        csv_path = settings.dashboard_uploads_dir / f"{metadata_path.stem}.csv"
        stored_hash = metadata.get("content_sha256")
        if stored_hash is None and csv_path.is_file():
            digest = hashlib.sha256()
            with csv_path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            stored_hash = digest.hexdigest()
        if stored_hash == content_sha256:
            if isinstance(dataset_id, str) and dataset_id.isalnum() and len(dataset_id) == 32:
                return dataset_id
    return None


def resolve_dataset_path(dataset_id: str | None = None) -> Path:
    """Resolve a catalog ID to the demo CSV or a validated dashboard upload."""
    selected_id = dataset_id or _active_id()
    if selected_id == DEMO_DATASET_ID:
        if not settings.sim_input_csv.is_file():
            raise FileNotFoundError("Demo prediction dataset is unavailable.")
        return settings.sim_input_csv

    if not selected_id.isalnum() or len(selected_id) != 32:
        raise FileNotFoundError(f"Dataset '{selected_id}' was not found.")
    csv_path = settings.dashboard_uploads_dir / f"{selected_id}.csv"
    metadata_path = settings.dashboard_uploads_dir / f"{selected_id}.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Dataset '{selected_id}' was not found.")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Dataset '{selected_id}' metadata is unreadable: {exc}") from exc
    if metadata.get("source") == "local":
        local_path = Path(str(metadata.get("local_path", ""))).resolve()
        if local_path != settings.raw_bitbrains_csv_path:
            raise FileNotFoundError(
                f"Local dataset '{selected_id}' no longer matches RAW_BITBRAINS_CSV."
            )
        if not local_path.is_file():
            raise FileNotFoundError(f"Configured local CSV is unavailable: {local_path}")
        stat = local_path.stat()
        if (
            metadata.get("local_size_bytes") != stat.st_size
            or metadata.get("local_modified_at_ns") != stat.st_mtime_ns
            or not _local_file_matches_metadata(metadata, stat)
        ):
            raise RuntimeError(
                "The configured local CSV changed after validation; validate it again "
                "before running inference."
            )
        if metadata.get("validation", {}).get("valid") is not True:
            raise RuntimeError(f"Dataset '{selected_id}' is not validated.")
        return local_path
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset '{selected_id}' was not found.")
    if metadata.get("validation", {}).get("valid") is not True:
        raise RuntimeError(f"Dataset '{selected_id}' is not validated.")
    return csv_path


def get_dataset_metadata_for_id(dataset_id: str) -> dict[str, Any]:
    if dataset_id == DEMO_DATASET_ID:
        return {
            "dataset_id": DEMO_DATASET_ID,
            "filename": settings.sim_input_csv.name,
            "dataset_kind": "prediction_csv",
        }
    if not dataset_id.isalnum() or len(dataset_id) != 32:
        raise FileNotFoundError(f"Dataset '{dataset_id}' was not found.")
    metadata_path = settings.dashboard_uploads_dir / f"{dataset_id}.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Dataset '{dataset_id}' was not found.")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Dataset '{dataset_id}' metadata is unreadable: {exc}") from exc
    return metadata


def inference_predictions_path(dataset_id: str) -> Path:
    return settings.dashboard_runs_dir / "inference" / dataset_id / "predictions.csv"


@lru_cache(maxsize=1)
def _read_prediction_frame(
    path_string: str,
    modified_at: int,
    size_bytes: int,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return pd.read_csv(path_string, usecols=list(columns))


def read_prediction_frame(
    dataset_id: str | None = None,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Read predictions from the selected input or its isolated inference output."""
    path: Path
    if run_id:
        from ..runner import get_run

        run = get_run(run_id)
        if run is None:
            raise FileNotFoundError(f"Run '{run_id}' was not found.")
        if run.get("dataset_id") != (dataset_id or get_active_dataset_id()):
            raise ValueError("The selected run belongs to a different dataset.")
        if run.get("data_mode") == "raw_workload_csv":
            if run.get("status") != "COMPLETED":
                raise ValueError("Raw workload forecasts are available after run completion.")
            path = settings.dashboard_runs_dir / run_id / "selected_predictions.csv"
        else:
            path = resolve_dataset_path(dataset_id)
    else:
        selected_id = dataset_id or get_active_dataset_id()
        dataset_kind = get_dataset_metadata_for_id(selected_id).get("dataset_kind")
        if dataset_kind == "raw_workload_csv":
            path = inference_predictions_path(selected_id)
            if not path.is_file():
                return pd.DataFrame(columns=PREDICTION_COLUMNS)
        elif dataset_kind == "incompatible_raw_workload_csv":
            return pd.DataFrame(columns=PREDICTION_COLUMNS)
        else:
            path = resolve_dataset_path(dataset_id)
    stat = path.stat()
    header = pd.read_csv(path, nrows=0)
    available_columns = tuple(column for column in PREDICTION_COLUMNS if column in header.columns)
    return _read_prediction_frame(str(path), stat.st_mtime_ns, stat.st_size, available_columns)


def _catalog_entry(
    dataset_id: str,
    filename: str,
    path: Path,
    source: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    if source == "demo":
        info = input_csv_metadata(path)
        if not info["exists"]:
            return {
                "dataset_id": dataset_id,
                "filename": filename,
                "source": source,
                "dataset_kind": "prediction_csv",
                "created_at": created_at,
                "size_bytes": 0,
                "row_count": 0,
                "vm_count": 0,
                "columns": [],
                "validation": {
                    "valid": False,
                    "errors": ["Demo CSV is unavailable."],
                    "warnings": [],
                    "missing_values": {},
                },
                "preview": [],
            }
        frame = pd.DataFrame(info["sample_rows"])
        validation = {
                **info["validation"],
                "row_count": info["total_rows"],
                "vm_count": info.get("vm_count", 0),
                "columns": list(frame.columns),
                "optional_columns_present": [
                    column for column in OPTIONAL_COLUMNS if column in frame.columns
                ],
        }
        preview = info["sample_rows"]
        size_bytes = path.stat().st_size
        dataset_kind = "prediction_csv"
    elif source == "local":
        metadata_path = settings.dashboard_uploads_dir / f"{dataset_id}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        validation = metadata["validation"]
        preview = metadata.get("preview", [])
        size_bytes = metadata["size_bytes"]
        dataset_kind = metadata.get("dataset_kind", "raw_workload_csv")
    else:
        metadata_path = settings.dashboard_uploads_dir / f"{dataset_id}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        validation = metadata["validation"]
        preview = metadata["preview"]
        size_bytes = metadata["size_bytes"]
        dataset_kind = metadata.get("dataset_kind", "prediction_csv")
    return {
        "dataset_id": dataset_id,
        "filename": filename,
        "source": source,
        "dataset_kind": dataset_kind,
        "created_at": created_at,
        "size_bytes": size_bytes,
        "row_count": validation.get("row_count", 0),
        "vm_count": validation.get("vm_count", 0),
        "vm_ids": validation.get("vm_ids", []),
        "columns": validation.get("columns", []),
        "validation": validation,
        "preview": preview,
    }


def list_datasets() -> list[dict[str, Any]]:
    datasets = [
        _catalog_entry(
            DEMO_DATASET_ID,
            settings.sim_input_csv.name,
            settings.sim_input_csv,
            "demo",
        )
    ]
    for metadata_path in settings.dashboard_uploads_dir.glob("*.json"):
        if metadata_path.name == ACTIVE_DATASET_FILE.name:
            continue
        dataset_id = metadata_path.stem
        if not dataset_id.isalnum() or len(dataset_id) != 32:
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            source = metadata.get("source", "upload")
            csv_path = settings.dashboard_uploads_dir / f"{dataset_id}.csv"
            if source == "local":
                local_path = Path(str(metadata.get("local_path", "")))
                if not local_path.is_file():
                    continue
                local_stat = local_path.stat()
                if (
                    metadata.get("local_size_bytes") != local_stat.st_size
                    or metadata.get("local_modified_at_ns") != local_stat.st_mtime_ns
                    or not _local_file_matches_metadata(metadata, local_stat)
                ):
                    continue
                entry_path = local_path
            else:
                if not csv_path.is_file():
                    continue
                entry_path = csv_path
            datasets.append(
                _catalog_entry(
                    dataset_id,
                    metadata["filename"],
                    entry_path,
                    source,
                    metadata.get("created_at"),
                )
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f"Unable to read dataset metadata '{dataset_id}': {exc}") from exc
    return sorted(datasets, key=lambda item: item.get("created_at") or "", reverse=True)


@router.get("/dataset/metadata")
def get_dataset_metadata() -> dict[str, Any]:
    datasets = list_datasets()
    active_id = _active_id()
    active = next((item for item in datasets if item["dataset_id"] == active_id), None)
    if active is None:
        raise HTTPException(status_code=409, detail="The selected dataset is no longer available.")
    inference = model_inference_status()
    return {
        "active_dataset_id": active_id,
        "active_dataset": active,
        "datasets": datasets,
        "upload_max_bytes": MAX_UPLOAD_BYTES,
        "simulation_note": (
            "Uploaded data can be simulated only from an isolated simulator copy. "
            "The frozen original input and result directories are never changed."
        ),
        "model_inference": inference,
        "model_inference_available": inference["available"],
        "inference_note": (
            "Raw inference is unavailable: "
            + " ".join(inference["missing"])
            if inference["missing"]
            else "Raw PatchTST forecasts and Logistic Regression risk scores are available."
        ),
    }


@router.post("/dataset/local/validate")
async def validate_configured_local_csv() -> dict[str, Any]:
    """Validate the configured large raw CSV in place, without browser upload."""
    path = settings.raw_bitbrains_csv_path.resolve()
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Configured RAW_BITBRAINS_CSV is unavailable: {path}",
        )
    stat = path.stat()
    identity = f"{path}|{stat.st_size}|{stat.st_mtime_ns}"
    dataset_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
    metadata_path = settings.dashboard_uploads_dir / f"{dataset_id}.json"
    if metadata_path.is_file():
        try:
            cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Local dataset validation metadata is unreadable: {exc}",
            ) from exc
        if (
            all(key in cached_metadata for key in _LOCAL_FILE_IDENTITY_FIELDS)
            and not _local_file_matches_metadata(cached_metadata, stat)
        ):
            identity = (
                f"{path}|{stat.st_size}|{stat.st_mtime_ns}|{stat.st_ctime_ns}|"
                f"{stat.st_dev}|{stat.st_ino}"
            )
            dataset_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
            metadata_path = settings.dashboard_uploads_dir / f"{dataset_id}.json"
            if metadata_path.is_file():
                try:
                    cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Local dataset validation metadata is unreadable: {exc}",
                    ) from exc
            else:
                cached_metadata = {}
        if (
            cached_metadata.get("source") == "local"
            and cached_metadata.get("local_path") == str(path)
            and cached_metadata.get("local_size_bytes") == stat.st_size
            and cached_metadata.get("local_modified_at_ns") == stat.st_mtime_ns
            and _local_file_matches_metadata(cached_metadata, stat)
            and all(key in cached_metadata for key in _LOCAL_FILE_IDENTITY_FIELDS)
            and cached_metadata.get("validation", {}).get("valid") is True
            and isinstance(cached_metadata.get("validation", {}).get("vm_ids"), list)
        ):
            entry = _catalog_entry(dataset_id, path.name, path, "local", cached_metadata.get("created_at"))
            return {
                "dataset": entry,
                "active_dataset_id": _active_id(),
                "validation_duration_seconds": cached_metadata.get(
                    "validation_duration_seconds", 0
                ),
                "validation_cached": True,
                "format_note": (
                    "The unchanged local raw Bitbrains CSV was previously validated. "
                    "Fresh inference uses the configured saved scaler, PatchTST checkpoint, "
                    "and Logistic Regression classifier."
                ),
            }
    validation_started = time.perf_counter()
    try:
        validation = await run_in_threadpool(validate_raw_bitbrains_file, path)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    validation_duration = time.perf_counter() - validation_started
    final_stat = path.stat()
    if (
        final_stat.st_size != stat.st_size
        or final_stat.st_mtime_ns != stat.st_mtime_ns
        or final_stat.st_ctime_ns != stat.st_ctime_ns
        or final_stat.st_dev != stat.st_dev
        or final_stat.st_ino != stat.st_ino
    ):
        raise HTTPException(
            status_code=409,
            detail="The configured local CSV changed during validation; retry after the file is stable.",
        )
    if not validation["valid"]:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Configured local raw Bitbrains CSV failed validation.",
                **validation,
                "validation_duration_seconds": round(validation_duration, 3),
            },
        )

    created_at = pd.Timestamp.utcnow().isoformat()
    metadata = {
        "dataset_id": dataset_id,
        "filename": path.name,
        "created_at": created_at,
        "size_bytes": stat.st_size,
        "local_path": str(path),
        "local_size_bytes": stat.st_size,
        "local_modified_at_ns": stat.st_mtime_ns,
        **_local_file_identity(stat),
        "source": "local",
        "validation_duration_seconds": round(validation_duration, 3),
        "validation": validation,
        "preview": validation.get("preview", []),
        "dataset_kind": "raw_workload_csv",
    }
    try:
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to save local dataset validation metadata: {exc}",
        ) from exc
    entry = _catalog_entry(dataset_id, path.name, path, "local", created_at)
    return {
        "dataset": entry,
        "active_dataset_id": _active_id(),
        "validation_duration_seconds": round(validation_duration, 3),
        "validation_cached": False,
        "format_note": (
            "Local raw Bitbrains CSV validated in chunks. Fresh inference uses the "
            "configured saved scaler, PatchTST checkpoint, and Logistic Regression classifier."
        ),
    }


@router.post("/dataset/select")
def select_dataset(request: DatasetSelection) -> dict[str, Any]:
    try:
        resolve_dataset_path(request.dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    entry = next(
        (item for item in list_datasets() if item["dataset_id"] == request.dataset_id),
        None,
    )
    if entry is None:
        raise HTTPException(status_code=404, detail="Dataset metadata is unavailable.")
    _set_active_id(request.dataset_id)
    return {"active_dataset": entry, "selected": True}


@router.post("/dataset/infer")
async def infer_dataset(request: DatasetSelection) -> dict[str, Any]:
    """Generate fresh model outputs only; this endpoint does not launch CloudSim."""
    try:
        metadata = get_dataset_metadata_for_id(request.dataset_id)
        input_path = resolve_dataset_path(request.dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if metadata.get("dataset_kind") != "raw_workload_csv":
        raise HTTPException(
            status_code=422,
            detail=(
                "Fresh model inference requires a compatible raw Bitbrains workload "
                "dataset. Prepared predictions remain recorded inputs."
            ),
        )

    inference = model_inference_status()
    if not inference["available"]:
        raise HTTPException(
            status_code=503,
            detail="Model inference artifacts are unavailable: " + " ".join(inference["missing"]),
        )

    output_path = inference_predictions_path(request.dataset_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f"predictions-{uuid.uuid4().hex}.tmp.csv"
    )
    try:
        result = await run_in_threadpool(
            infer_and_merge_dataset,
            input_path,
            output_path,
            temporary_path,
            str(metadata.get("filename") or input_path.name),
            request.vm_ids,
            request.start_time,
            request.end_time,
        )
    except HTTPException:
        temporary_path.unlink(missing_ok=True)
        raise
    except (
        OSError,
        ValueError,
        RuntimeError,
        pd.errors.ParserError,
    ) as exc:
        temporary_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=f"Fresh inference failed: {exc}",
        ) from exc

    return {
        "dataset_id": request.dataset_id,
        "data_mode": "fresh_model_inference",
        "row_count": result["row_count"],
        "vm_count": result["vm_count"],
        "total_prediction_rows": result["total_row_count"],
        "completed_vm_ids": result["completed_vm_ids"],
        "completed_vm_count": result["completed_vm_count"],
        "processed_vm_count": result["processed_vm_count"],
        "checkpoint_prediction_rows": result["checkpoint_row_count"],
        "risk_threshold": inference["risk_threshold"],
        "message": (
            "Fresh PatchTST forecasts and Logistic Regression risk scores were generated "
            "for the requested VM batch and merged into resumable per-dataset inference output."
        ),
    }


@router.post("/dataset/validate")
async def validate_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )
    validation_started = time.perf_counter()
    validation = await run_in_threadpool(validate_upload_csv, content, file.filename)
    validation_duration = time.perf_counter() - validation_started
    columns = set(validation.get("columns", []))
    dataset_kind = validation.get("dataset_kind", "raw_or_incompatible_csv")
    missing_simulator_fields = [
        column for column in REQUIRED_COLUMNS if column not in columns
    ]
    format_note = upload_compatibility_note(
        dataset_kind,
        list(validation.get("columns", [])),
        file.filename,
    )
    content_sha256 = hashlib.sha256(content).hexdigest()
    duplicate_dataset_id = _duplicate_dataset_id(content_sha256)
    preview = validation.get("preview", [])
    return {
        "filename": Path(file.filename or "dataset.csv").name,
        "size_bytes": len(content),
        "validation": validation,
        "dataset_kind": dataset_kind,
        "format_note": format_note,
        "inference_available": model_inference_status()["available"],
        "content_sha256": content_sha256,
        "validation_duration_seconds": round(validation_duration, 3),
        "duplicate_dataset_id": duplicate_dataset_id,
        "preview": preview,
    }


@router.post("/dataset/upload", status_code=201)
async def upload_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )
    validation_started = time.perf_counter()
    validation = await run_in_threadpool(validate_upload_csv, content, file.filename)
    validation_duration = time.perf_counter() - validation_started
    if not validation["valid"]:
        dataset_kind = validation.get("dataset_kind", "raw_or_incompatible_csv")
        format_note = upload_compatibility_note(
            dataset_kind,
            list(validation.get("columns", [])),
            file.filename,
        )
        raise HTTPException(
            status_code=422,
            detail={
                "message": format_note,
                "validation_message": "CSV validation failed; the file was not stored.",
                "format_note": format_note,
                **validation,
            },
        )

    dataset_kind = validation.get("dataset_kind", "raw_or_incompatible_csv")
    if dataset_kind not in (
        "prediction_csv",
        "raw_workload_csv",
        "incompatible_raw_workload_csv",
    ):
        format_note = upload_compatibility_note(
            dataset_kind,
            list(validation.get("columns", [])),
            file.filename,
        )
        raise HTTPException(
            status_code=422,
            detail={
                "message": format_note,
                "format_note": format_note,
                **validation,
            },
        )
    if dataset_kind == "prediction_csv" and not validation.get("simulator_compatible"):
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Prepared prediction CSV is not compatible with CloudSim.",
                **validation,
            },
        )

    content_sha256 = hashlib.sha256(content).hexdigest()
    duplicate_dataset_id = _duplicate_dataset_id(content_sha256)
    if duplicate_dataset_id:
        existing = next(
            (
                item
                for item in list_datasets()
                if item["dataset_id"] == duplicate_dataset_id
            ),
            None,
        )
        if existing is None:
            raise HTTPException(
                status_code=409,
                detail="Duplicate dataset metadata is unavailable; refresh the dataset catalog.",
            )
        return {
            "dataset": existing,
            "active_dataset_id": _active_id(),
            "activated": False,
            "duplicate": True,
        }

    dataset_id = uuid.uuid4().hex
    filename = Path(file.filename or "dataset.csv").name
    created_at = pd.Timestamp.utcnow().isoformat()
    preview = validation.get("preview", [])
    metadata = {
        "dataset_id": dataset_id,
        "filename": filename,
        "created_at": created_at,
        "size_bytes": len(content),
        "content_sha256": content_sha256,
        "validation_duration_seconds": round(validation_duration, 3),
        "validation": validation,
        "preview": preview,
        "dataset_kind": dataset_kind,
    }

    csv_path = settings.dashboard_uploads_dir / f"{dataset_id}.csv"
    metadata_path = settings.dashboard_uploads_dir / f"{dataset_id}.json"
    try:
        csv_path.write_bytes(content)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    except OSError as exc:
        csv_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Unable to store validated dataset: {exc}") from exc
    return {
        "dataset": _catalog_entry(dataset_id, filename, csv_path, "upload", created_at),
        "active_dataset_id": _active_id(),
        "activated": False,
        "duplicate": False,
        "validation_duration_seconds": round(validation_duration, 3),
        "simulation_note": (
            "This dataset is stored but not active. Select it explicitly to use it. Runs use a private simulator copy "
            "inside the dashboard's run directory."
        ),
    }
