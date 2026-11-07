"""Training-compatible PatchTST forecasts and adaptive risk probabilities."""
from __future__ import annotations

import csv
import logging
import pickle
import sqlite3
import tempfile
import warnings
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import settings
from .data import RAW_RESOURCE_COLUMNS, RAW_REQUIRED_COLUMNS

logger = logging.getLogger(__name__)

FEATURES = [
    "CPU usage [%]",
    "Memory usage [KB]",
    "Disk read throughput [KB/s]",
    "Disk write throughput [KB/s]",
]
LOOKBACK = 96
HORIZON = 12
EXPORT_INTERVAL_STEPS = 3
WINDOW_BATCH_SIZE = 256
CLASSIFIER_FEATURE_COUNT = 48
EXPECTED_SKLEARN_VERSION = settings.classifier_sklearn_version
OUTPUT_COLUMNS = [
    "vm_id",
    "decision_time",
    "forecast_start",
    "forecast_end",
    "current_cpu_pct",
    "current_memory_kb",
    "current_disk_read_kbps",
    "current_disk_write_kbps",
    "risk_probability",
    "risk_threshold",
    "risk_alert",
    "forecast_cpu_max",
    "forecast_cpu_mean",
    "forecast_memory_max",
    "forecast_memory_mean",
    "forecast_disk_read_max",
    "forecast_disk_read_mean",
    "forecast_disk_write_max",
    "forecast_disk_write_mean",
]


class InferenceArtifactError(RuntimeError):
    """A required trained inference artifact is absent or incompatible."""


def _sklearn_version() -> str | None:
    try:
        return version("scikit-learn")
    except PackageNotFoundError:
        return None


def _artifact_preflight() -> tuple[list[str], str | None, str | None]:
    missing: list[str] = []
    paths = {
        "patchtst_checkpoint": settings.patchtst_checkpoint_path,
        "train_only_scaler": settings.scaler_path,
        "logistic_regression_classifier": settings.lr_classifier_path,
    }
    for name, path in paths.items():
        if not path.is_file():
            missing.append(f"{name} is unavailable at {path}.")

    installed_sklearn = _sklearn_version()
    if installed_sklearn is None:
        missing.append("scikit-learn is not installed in the backend environment.")
    elif installed_sklearn != EXPECTED_SKLEARN_VERSION:
        missing.append(
            "The classifier was saved with scikit-learn "
            f"{EXPECTED_SKLEARN_VERSION}; installed version is {installed_sklearn}."
        )

    try:
        torch_version = version("torch")
    except PackageNotFoundError:
        torch_version = None
        missing.append("PyTorch is not installed in the backend environment.")

    return missing, installed_sklearn, torch_version


def model_inference_status() -> dict[str, Any]:
    missing, installed_sklearn, torch_version = _artifact_preflight()
    if not missing:
        try:
            _load_artifacts()
        except InferenceArtifactError as exc:
            missing.append(str(exc))

    paths = {
        "patchtst_checkpoint": settings.patchtst_checkpoint_path,
        "train_only_scaler": settings.scaler_path,
        "logistic_regression_classifier": settings.lr_classifier_path,
    }
    return {
        "available": not missing,
        "missing": missing,
        "artifacts": {name: str(path) for name, path in paths.items()},
        "sklearn_version": installed_sklearn,
        "torch_version": torch_version,
        "feature_order": FEATURES,
        "lookback_steps": LOOKBACK,
        "forecast_horizon_steps": HORIZON,
        "classifier_feature_count": CLASSIFIER_FEATURE_COUNT,
        "risk_threshold": settings.lr_risk_threshold,
    }


@lru_cache(maxsize=1)
def _load_artifacts() -> tuple[Any, Any, Any, Any, Any]:
    missing, _, _ = _artifact_preflight()
    if missing:
        raise InferenceArtifactError(
            "Raw workload inference is unavailable: " + " ".join(missing)
        )

    try:
        import joblib
        import torch
        import torch.nn as nn
        from sklearn.exceptions import InconsistentVersionWarning
    except (ImportError, OSError) as exc:
        raise InferenceArtifactError(
            f"Unable to import the inference runtime: {exc}"
        ) from exc

    class PatchTST(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            seq_len, pred_len, channels = LOOKBACK, HORIZON, len(FEATURES)
            patch_len, stride, d_model = 16, 8, 128
            n_patches = (seq_len - patch_len) // stride + 1
            self.channels = channels
            self.patch_len = patch_len
            self.stride = stride
            self.revin = False
            self.patch_proj = nn.Linear(patch_len, d_model)
            self.pos = nn.Parameter(torch.zeros(1, n_patches, d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=256,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
            self.head = nn.Linear(n_patches * d_model, pred_len)
            nn.init.trunc_normal_(self.pos, std=0.02)

        def forward(self, values: Any) -> Any:
            batch, length, channels = values.shape
            sequence = values.transpose(1, 2).reshape(batch * channels, length)
            sequence = sequence.unfold(1, self.patch_len, self.stride)
            sequence = self.patch_proj(sequence)
            sequence = self.encoder(sequence + self.pos[:, : sequence.size(1), :])
            sequence = self.head(sequence.flatten(1))
            return sequence.reshape(batch, channels, HORIZON).transpose(1, 2)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            scaler = joblib.load(settings.scaler_path)
            classifier = joblib.load(settings.lr_classifier_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(
            settings.patchtst_checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except (
        OSError,
        EOFError,
        ValueError,
        TypeError,
        RuntimeError,
        pickle.UnpicklingError,
        InconsistentVersionWarning,
    ) as exc:
        raise InferenceArtifactError(
            f"Unable to load a compatible inference artifact: {exc}"
        ) from exc

    if getattr(scaler, "n_features_in_", None) != len(FEATURES):
        raise InferenceArtifactError(
            "The train-only scaler must be fitted to exactly four resources in "
            f"this order: {', '.join(FEATURES)}."
        )
    scaler_features = getattr(scaler, "feature_names_in_", None)
    if scaler_features is not None and list(scaler_features) != FEATURES:
        raise InferenceArtifactError(
            "The train-only scaler resource order does not match the model's required "
            f"feature order: {', '.join(FEATURES)}."
        )
    if getattr(classifier, "n_features_in_", None) != CLASSIFIER_FEATURE_COUNT:
        raise InferenceArtifactError(
            "The Logistic Regression artifact must accept the notebook's "
            f"{CLASSIFIER_FEATURE_COUNT}-feature risk vector."
        )
    classifier_classes = getattr(classifier, "classes_", None)
    if (
        not callable(getattr(classifier, "predict_proba", None))
        or classifier_classes is None
        or 1 not in classifier_classes
    ):
        raise InferenceArtifactError(
            "The Logistic Regression artifact must expose probabilities for class 1."
        )

    model = PatchTST().to(device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise InferenceArtifactError(
            f"PatchTST checkpoint does not match the verified 96x4 to 12x4 "
            f"architecture: {exc}"
        ) from exc
    model.eval()
    try:
        with torch.inference_mode():
            output = model(
                torch.zeros(
                    (1, LOOKBACK, len(FEATURES)),
                    dtype=next(model.parameters()).dtype,
                    device=device,
                )
            )
        scaler.transform(np.zeros((1, len(FEATURES)), dtype=np.float64))
        scaler.inverse_transform(np.zeros((1, len(FEATURES)), dtype=np.float64))
        probability = classifier.predict_proba(
            np.zeros((1, CLASSIFIER_FEATURE_COUNT), dtype=np.float64)
        )
    except (ValueError, TypeError, RuntimeError, AttributeError) as exc:
        raise InferenceArtifactError(
            f"Loaded inference artifacts failed their compatibility check: {exc}"
        ) from exc
    if tuple(output.shape) != (1, HORIZON, len(FEATURES)):
        raise InferenceArtifactError(
            "PatchTST failed its 96x4 to 12x4 compatibility check: "
            f"received output shape {tuple(output.shape)}."
        )
    if probability.shape != (1, len(classifier_classes)) or not np.isfinite(probability).all():
        raise InferenceArtifactError(
            "The Logistic Regression artifact failed its probability compatibility check."
        )
    return model, scaler, classifier, device, torch


def _risk_features(history: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    hist_mean = history.mean(axis=1)
    hist_std = history.std(axis=1)
    hist_max = history.max(axis=1)
    hist_min = history.min(axis=1)
    hist_last = history[:, -1, :]
    hist_recent = history[:, -12:, :].mean(axis=1)
    hist_slope = (history[:, -1, :] - history[:, 0, :]) / 95.0

    pred_mean = prediction.mean(axis=1)
    pred_std = prediction.std(axis=1)
    pred_max = prediction.max(axis=1)
    pred_min = prediction.min(axis=1)
    pred_slope = prediction[:, -1, :] - prediction[:, 0, :]

    return np.concatenate(
        [
            hist_mean,
            hist_std,
            hist_max,
            hist_min,
            hist_last,
            hist_recent,
            hist_slope,
            pred_mean,
            pred_std,
            pred_max,
            pred_min,
            pred_slope,
        ],
        axis=1,
    ).astype(np.float32)


def _raw_csv_settings(path: Path) -> tuple[str, str | None]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header_line = stream.readline()
    delimiters = [",", ";"]
    delimiter = max(
        delimiters,
        key=lambda candidate: len(next(csv.reader([header_line], delimiter=candidate), [])),
    )
    headers = [
        str(column).strip().rstrip(";").strip()
        for column in next(csv.reader([header_line], delimiter=delimiter), [])
    ]
    vm_column = next(
        (column for column in ("VM_ID", "vm_id") if column in headers),
        None,
    )
    missing = [column for column in RAW_REQUIRED_COLUMNS if column not in headers]
    if missing:
        raise ValueError(
            "Raw workload CSV is missing training-schema columns: "
            + ", ".join(missing)
            + "."
        )
    return delimiter, vm_column


def _predict_trace_records(
    vm_id: str,
    trace: pd.DataFrame,
    scaler: Any,
    classifier: Any,
    class_index: int,
    model: Any,
    model_dtype: Any,
    torch: Any,
    on_stage: Callable[[str], None] | None,
    write_records: Callable[[list[dict[str, Any]]], None],
    start_time: pd.Timestamp | None,
    end_time: pd.Timestamp | None,
) -> int:
    trace = trace.copy()
    timestamp = pd.to_numeric(trace["Timestamp [ms]"], errors="coerce")
    if timestamp.isna().any() or not np.isfinite(timestamp.to_numpy()).all():
        raise ValueError("Raw 'Timestamp [ms]' values must be finite Unix seconds.")
    trace["Timestamp"] = pd.to_datetime(timestamp, unit="s", errors="coerce")
    for column in RAW_RESOURCE_COLUMNS:
        converted = pd.to_numeric(trace[column], errors="coerce")
        if (trace[column].notna() & converted.isna()).any():
            raise ValueError(f"Raw resource column '{column}' contains non-numeric values.")
        trace[column] = converted
        trace.loc[trace[column] < 0, column] = np.nan

    work = trace[["Timestamp", *RAW_RESOURCE_COLUMNS]].dropna(subset=["Timestamp"])
    work = (
        work.sort_values("Timestamp")
        .groupby("Timestamp", as_index=False)[RAW_RESOURCE_COLUMNS]
        .mean()
    )
    resampled = work.set_index("Timestamp")[RAW_RESOURCE_COLUMNS].resample("5min").mean()
    raw = resampled[FEATURES].to_numpy(dtype=np.float64)
    valid = np.isfinite(raw).all(axis=1)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) < LOOKBACK + HORIZON:
        return 0

    timestamps_ns = resampled.index.asi8
    breaks = np.where(
        (np.diff(valid_indices) != 1)
        | (np.diff(timestamps_ns[valid_indices]) != 300 * 10**9)
    )[0] + 1
    record_count = 0
    for segment_indices in np.split(valid_indices, breaks):
        if len(segment_indices) < LOOKBACK + HORIZON:
            continue
        values_raw = raw[segment_indices]
        values_scaled = scaler.transform(values_raw).astype(np.float32)
        segment_times = resampled.index[segment_indices]
        last_start = len(segment_indices) - LOOKBACK - HORIZON
        starts = np.arange(0, last_start + 1, EXPORT_INTERVAL_STEPS)
        decision_times = segment_times[starts + LOOKBACK - 1]
        if start_time is not None:
            starts = starts[decision_times >= start_time]
            decision_times = decision_times[decision_times >= start_time]
        if end_time is not None:
            starts = starts[decision_times <= end_time]

        for batch_start in range(0, len(starts), WINDOW_BATCH_SIZE):
            batch_starts = starts[batch_start : batch_start + WINDOW_BATCH_SIZE]
            histories_raw = np.stack(
                [values_raw[start : start + LOOKBACK] for start in batch_starts]
            )
            histories_scaled = np.stack(
                [values_scaled[start : start + LOOKBACK] for start in batch_starts]
            )
            if on_stage:
                on_stage("forecasting_and_risk_scoring")
            with torch.inference_mode():
                output = model(
                    torch.as_tensor(
                        histories_scaled,
                        dtype=model_dtype,
                        device=next(model.parameters()).device,
                    )
                )
            if tuple(output.shape[1:]) != (HORIZON, len(FEATURES)):
                raise InferenceArtifactError(
                    "PatchTST returned an unexpected output shape: "
                    f"{tuple(output.shape)}."
                )
            predictions_scaled = output.detach().cpu().numpy()
            if not np.isfinite(predictions_scaled).all():
                raise InferenceArtifactError("PatchTST returned non-finite forecast values.")
            predictions_raw = scaler.inverse_transform(
                predictions_scaled.reshape(-1, len(FEATURES))
            ).reshape(-1, HORIZON, len(FEATURES))
            risk_input = _risk_features(histories_raw, predictions_raw)
            if risk_input.shape[1] != CLASSIFIER_FEATURE_COUNT:
                raise InferenceArtifactError(
                    f"Generated {risk_input.shape[1]} risk features; expected "
                    f"{CLASSIFIER_FEATURE_COUNT}."
                )
            probabilities = classifier.predict_proba(risk_input)[:, class_index]
            if (
                not np.isfinite(probabilities).all()
                or (probabilities < 0).any()
                or (probabilities > 1).any()
            ):
                raise InferenceArtifactError(
                    "Logistic Regression returned invalid risk probabilities."
                )
            demand = np.maximum(predictions_raw, 0.0)
            batch_records: list[dict[str, Any]] = []
            for offset, start in enumerate(batch_starts):
                decision_time = segment_times[start + LOOKBACK - 1]
                current = histories_raw[offset, -1]
                prediction = demand[offset]
                probability = float(probabilities[offset])
                row: dict[str, Any] = {
                    "vm_id": vm_id,
                    "decision_time": decision_time,
                    "forecast_start": decision_time + pd.Timedelta(minutes=5),
                    "forecast_end": decision_time + pd.Timedelta(minutes=60),
                    "current_cpu_pct": float(current[0]),
                    "current_memory_kb": float(current[1]),
                    "current_disk_read_kbps": float(current[2]),
                    "current_disk_write_kbps": float(current[3]),
                    "risk_probability": probability,
                    "risk_threshold": settings.lr_risk_threshold,
                    "risk_alert": int(probability >= settings.lr_risk_threshold),
                }
                for index, name in enumerate(
                    ("cpu", "memory", "disk_read", "disk_write")
                ):
                    row[f"forecast_{name}_max"] = float(prediction[:, index].max())
                    row[f"forecast_{name}_mean"] = float(prediction[:, index].mean())
                batch_records.append(row)
            write_records(batch_records)
            record_count += len(batch_records)
    return record_count


def generate_prediction_csv(
    input_path: Path,
    output_path: Path,
    source_filename: str,
    on_stage: Callable[[str], None] | None = None,
    vm_ids: set[str] | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    checkpoint_path: Path | None = None,
    on_vm_complete: Callable[[int, str | None], None] | None = None,
) -> dict[str, int]:
    model, scaler, classifier, device, torch = _load_artifacts()
    if not 0 <= settings.lr_risk_threshold <= 1:
        raise InferenceArtifactError("LR_RISK_THRESHOLD must be within [0, 1].")

    if on_stage:
        on_stage("preprocessing")
    selected_start = pd.Timestamp(start_time) if start_time else None
    selected_end = pd.Timestamp(end_time) if end_time else None
    if (
        selected_start is not None
        and selected_end is not None
        and selected_start > selected_end
    ):
        raise ValueError("start_time must be earlier than or equal to end_time.")
    class_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
    model_dtype = next(model.parameters()).dtype
    delimiter, vm_column = _raw_csv_settings(input_path)
    id_column = vm_column or "__single_vm_id__"
    usecols = [*RAW_REQUIRED_COLUMNS]
    if vm_column:
        usecols.append(vm_column)
    pending_vm: str | None = None
    pending_parts: list[pd.DataFrame] = []
    completed_vm_ids: set[str] = set()
    processed_vm_ids: set[str] = set()
    vm_count = 0
    row_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    persistent_database = checkpoint_path is not None
    if checkpoint_path is None:
        temporary_database = tempfile.NamedTemporaryFile(
            prefix="patchx-inference-",
            suffix=".sqlite3",
            dir=output_path.parent,
            delete=False,
        )
        database_path = Path(temporary_database.name)
        temporary_database.close()
    else:
        database_path = checkpoint_path
        database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    column_sql = ", ".join(
        f'"{column}" TEXT'
        for column in OUTPUT_COLUMNS
        if column not in {"vm_id", "decision_time"}
    )
    connection.execute(
        f'CREATE TABLE IF NOT EXISTS predictions ("vm_id" TEXT, '
        f'"decision_time" TEXT, {column_sql}, '
        'PRIMARY KEY ("vm_id", "decision_time"))'
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS completed_vms "
        "(vm_id TEXT PRIMARY KEY, generated_rows INTEGER NOT NULL)"
    )
    checkpoint_completed_ids = {
        str(row[0]) for row in connection.execute("SELECT vm_id FROM completed_vms")
    }
    initial_completed_ids = set(checkpoint_completed_ids)
    insert_sql = (
        "INSERT OR REPLACE INTO predictions ("
        + ", ".join(f'"{column}"' for column in OUTPUT_COLUMNS)
        + ") VALUES ("
        + ", ".join("?" for _ in OUTPUT_COLUMNS)
        + ")"
    )
    if on_vm_complete:
        initial_completed = (
            initial_completed_ids
            if vm_ids is None
            else initial_completed_ids.intersection(vm_ids)
        )
        on_vm_complete(len(initial_completed), None)

    def write_records(batch: list[dict[str, Any]]) -> None:
        nonlocal row_count
        if not batch:
            return
        connection.executemany(
            insert_sql,
            [
                tuple(
                    (
                        value.strftime("%Y-%m-%d %H:%M:%S")
                        if isinstance(value, pd.Timestamp)
                        else str(value)
                    )
                    for value in (record[column] for column in OUTPUT_COLUMNS)
                )
                for record in batch
            ],
        )
        row_count += len(batch)

    def finish_vm() -> None:
        nonlocal pending_vm, pending_parts, vm_count
        if pending_vm is None:
            return
        trace = pd.concat(pending_parts, ignore_index=True)
        generated_rows = _predict_trace_records(
            pending_vm,
            trace,
            scaler,
            classifier,
            class_index,
            model,
            model_dtype,
            torch,
            on_stage,
            write_records,
            selected_start,
            selected_end,
        )
        connection.execute(
            "INSERT OR REPLACE INTO completed_vms (vm_id, generated_rows) VALUES (?, ?)",
            (pending_vm, generated_rows),
        )
        connection.commit()
        if generated_rows:
            completed_vm_ids.add(pending_vm)
            vm_count += 1
        checkpoint_completed_ids.add(pending_vm)
        if on_vm_complete:
            completed_count = (
                len(checkpoint_completed_ids)
                if vm_ids is None
                else len(checkpoint_completed_ids.intersection(vm_ids))
            )
            on_vm_complete(completed_count, pending_vm)
        pending_vm = None
        pending_parts = []

    try:
        chunks = pd.read_csv(
            input_path,
            sep=delimiter,
            usecols=usecols,
            chunksize=50_000,
            encoding="utf-8-sig",
        )
        for chunk in chunks:
            chunk.columns = [
                str(column).strip().rstrip(";").strip()
                for column in chunk.columns
            ]
            if vm_column is None:
                chunk[id_column] = "1"
            if chunk[id_column].isna().any():
                raise ValueError(f"Raw workload column '{id_column}' contains empty VM IDs.")
            identifiers = chunk[id_column].astype(str).str.strip()
            if identifiers.eq("").any():
                raise ValueError(f"Raw workload column '{id_column}' contains empty VM IDs.")
            chunk[id_column] = identifiers
            group_ids = identifiers.ne(identifiers.shift()).cumsum()
            for _, group in chunk.groupby(group_ids, sort=False):
                vm_id = str(group[id_column].iloc[0])
                if vm_id == pending_vm:
                    pending_parts.append(group.drop(columns=id_column))
                    continue
                finish_vm()
                if vm_id in processed_vm_ids:
                    raise ValueError(
                        "Rows for each VM must be contiguous in the combined raw CSV "
                        "for bounded-memory inference."
                    )
                if vm_ids is not None and vm_id not in vm_ids:
                    continue
                processed_vm_ids.add(vm_id)
                if persistent_database and vm_id in checkpoint_completed_ids:
                    continue
                pending_vm = vm_id
                pending_parts.append(group.drop(columns=id_column))
        finish_vm()
        missing_vm_ids = (vm_ids or set()) - processed_vm_ids
        if missing_vm_ids:
            raise ValueError(
                "Requested VM IDs were not found in the raw CSV: "
                + ", ".join(sorted(missing_vm_ids))
                + "."
            )
        total_rows = int(
            connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        )
        total_vm_count = int(
            connection.execute(
                'SELECT COUNT(DISTINCT "vm_id") FROM predictions'
            ).fetchone()[0]
        )
        if total_rows == 0:
            raise ValueError(
                f"No continuous 5-minute workload segment contains at least "
                f"{LOOKBACK + HORIZON} rows for inference."
            )
        if on_stage:
            on_stage("writing_predictions")
        connection.commit()
        with output_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(OUTPUT_COLUMNS)
            cursor = connection.execute(
                'SELECT * FROM predictions ORDER BY "decision_time", "vm_id"'
            )
            while rows := cursor.fetchmany(10_000):
                writer.writerows(rows)
    except (OSError, UnicodeDecodeError, ValueError, sqlite3.Error, pd.errors.ParserError) as exc:
        raise ValueError(f"Cannot process raw workload CSV: {exc}") from exc
    finally:
        connection.close()
        if not persistent_database:
            try:
                database_path.unlink(missing_ok=True)
            except OSError:
                logger.exception("Unable to remove temporary inference database %s", database_path)

    logger.info(
        "Generated %s fresh PatchTST/classifier rows for %s VMs at %s",
        row_count,
        len(completed_vm_ids),
        output_path,
    )
    return {
        "row_count": row_count,
        "vm_count": vm_count,
        "total_row_count": total_rows,
        "total_vm_count": total_vm_count,
        "processed_vm_count": len(checkpoint_completed_ids),
        "resumed_vm_count": len(
            initial_completed_ids
            if vm_ids is None
            else initial_completed_ids.intersection(vm_ids)
        ),
    }
