"""Run the frozen Java simulator in a private, per-run project copy."""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import SUPPORTED_POLICIES, settings
from .data import simulator_compatibility_errors
from .routers.dataset import (
    get_active_dataset_id,
    get_dataset_metadata_for_id,
    infer_and_merge_dataset,
    inference_predictions_path,
    resolve_dataset_path,
)

logger = logging.getLogger(__name__)
_run_lock = threading.Lock()
_active_run_id: str | None = None
_runs: dict[str, dict[str, Any]] = {}
MAX_LOG_BYTES = 256 * 1024


class SimulatorInputError(ValueError):
    """The dataset does not satisfy the frozen simulator's CSV reader."""


def _meta_path(run_id: str) -> Path:
    return settings.dashboard_runs_dir / run_id / "meta.json"


def _save_meta(run_id: str) -> None:
    path = _meta_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_runs[run_id], indent=2, default=str), encoding="utf-8")


def _load_all_runs() -> None:
    for meta_file in sorted(settings.dashboard_runs_dir.glob("*/meta.json")):
        try:
            metadata = json.loads(meta_file.read_text(encoding="utf-8"))
            run_id = metadata.get("run_id")
            if not isinstance(run_id, str) or not run_id.startswith("run_"):
                logger.warning("Ignoring run metadata with an invalid run ID: %s", meta_file.name)
                continue
            if metadata.get("status") in ("RUNNING", "PENDING"):
                metadata["status"] = "INTERRUPTED"
                metadata["stage"] = "interrupted"
                metadata["error"] = "Backend restarted before the run completed."
                metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
                _runs[run_id] = metadata
                _save_meta(run_id)
            else:
                _runs[run_id] = metadata
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Unable to load run metadata from %s: %s", meta_file.name, exc)


def _load_run_metadata(run_id: str) -> dict[str, Any] | None:
    if not run_id.startswith("run_"):
        return None
    try:
        metadata = json.loads(_meta_path(run_id).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Unable to load run metadata for %s: %s", run_id, exc)
        return None
    if not isinstance(metadata, dict) or metadata.get("run_id") != run_id:
        return None
    _runs[run_id] = metadata
    return metadata


def list_runs() -> list[dict[str, Any]]:
    for meta_file in settings.dashboard_runs_dir.glob("*/meta.json"):
        run_id = meta_file.parent.name
        if run_id not in _runs:
            _load_run_metadata(run_id)
    return sorted(_runs.values(), key=lambda run: run.get("created_at", ""), reverse=True)


def get_run(run_id: str) -> dict[str, Any] | None:
    return _runs.get(run_id) or _load_run_metadata(run_id)


def _read_log(path: Path) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - MAX_LOG_BYTES))
        content = stream.read().decode("utf-8", errors="replace")
    if size > MAX_LOG_BYTES:
        return "[Earlier output omitted]\n" + content
    return content


def get_run_logs(run_id: str) -> dict[str, str]:
    run_dir = settings.dashboard_runs_dir / run_id
    return {
        "stdout": _read_log(run_dir / "stdout.log"),
        "stderr": _read_log(run_dir / "stderr.log"),
    }


def _prepare_isolated_project(
    run_id: str,
    dataset_id: str,
    input_path: Path | None = None,
) -> tuple[Path, Path]:
    source = settings.sim_project_dir.resolve()
    if not (source / "pom.xml").is_file() or not (source / "src").is_dir():
        raise FileNotFoundError("Configured simulator project is missing pom.xml or src/.")
    selected_input = input_path or resolve_dataset_path(dataset_id)
    run_dir = settings.dashboard_runs_dir / run_id
    project_copy = run_dir / "simulator"
    project_copy.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source / "pom.xml", project_copy / "pom.xml")
    shutil.copytree(source / "src", project_copy / "src")
    maven_config = source / ".mvn"
    if maven_config.is_dir():
        shutil.copytree(maven_config, project_copy / ".mvn")
    input_copy = project_copy / "data" / "bitbrains_cloudsim_predictions.csv"
    input_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_input, input_copy)
    simulator_errors = simulator_compatibility_errors(input_copy)
    if simulator_errors:
        raise SimulatorInputError("Simulator input is incompatible: " + " ".join(simulator_errors))
    output_dir = settings.run_output_dir(run_id)
    output_dir.mkdir(parents=True, exist_ok=False)
    return project_copy, output_dir


def _selection_datetime(value: str | None, field_name: str) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO local datetime.") from exc
    if parsed.tzinfo is not None:
        raise ValueError(f"{field_name} must not include a timezone.")
    return parsed


def _prepare_selected_predictions(
    input_path: Path,
    output_path: Path,
    vm_ids: list[str] | None,
    start_time: str | None,
    end_time: str | None,
    risk_threshold: float,
) -> dict[str, int]:
    selected_ids = set(vm_ids or [])
    start = _selection_datetime(start_time, "start_time")
    end = _selection_datetime(end_time, "end_time")
    if start and end and start > end:
        raise ValueError("start_time must be earlier than or equal to end_time.")

    row_count = 0
    selected_vms: set[str] = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not {
            "vm_id", "decision_time", "risk_probability", "risk_threshold"
        }.issubset(reader.fieldnames):
            raise SimulatorInputError("Prediction input is missing required simulation columns.")
        fieldnames = reader.fieldnames
        with output_path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                vm_id = str(row.get("vm_id", "")).strip()
                if selected_ids and vm_id not in selected_ids:
                    continue
                timestamp_value = str(row.get("decision_time", ""))
                try:
                    timestamp = datetime.fromisoformat(timestamp_value)
                    probability = float(row["risk_probability"])
                except (TypeError, ValueError, KeyError) as exc:
                    raise SimulatorInputError(
                        f"Invalid prediction row for VM '{vm_id}' at '{timestamp_value}'."
                    ) from exc
                if timestamp.tzinfo is not None:
                    raise SimulatorInputError(
                        "Prediction timestamps must be timezone-free local datetimes."
                    )
                if start and timestamp < start or end and timestamp > end:
                    continue
                row["risk_threshold"] = str(risk_threshold)
                if "risk_alert" in row:
                    row["risk_alert"] = str(int(probability >= risk_threshold))
                writer.writerow(row)
                selected_vms.add(vm_id)
                row_count += 1

    if selected_ids:
        missing_ids = selected_ids - selected_vms
        if missing_ids:
            raise ValueError(
                "No selected prediction rows were found for VM IDs: "
                + ", ".join(sorted(missing_ids))
                + "."
            )
    if row_count == 0:
        raise ValueError("The selected VM and time window contains no prediction rows.")
    return {"row_count": row_count, "vm_count": len(selected_vms)}


def _run_simulation_thread(
    run_id: str,
    policy: str,
    extra_props: dict[str, str],
    dataset_id: str,
    dataset_kind: str,
    source_filename: str,
    vm_ids: list[str] | None,
    start_time: str | None,
    end_time: str | None,
    risk_threshold: float,
) -> None:
    global _active_run_id
    run = _runs[run_id]
    run_dir = settings.dashboard_runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    stage_started = time.perf_counter()

    def set_stage(stage: str) -> None:
        nonlocal stage_started
        if run.get("stage") == stage:
            return
        elapsed = time.perf_counter() - stage_started
        durations = run.setdefault("stage_durations_seconds", {})
        previous = str(run.get("stage", "unknown"))
        durations[previous] = round(durations.get(previous, 0.0) + elapsed, 3)
        run["stage"] = stage
        stage_started = time.perf_counter()
        _save_meta(run_id)

    try:
        run["status"] = "RUNNING"
        run["started_at"] = datetime.now(timezone.utc).isoformat()
        _save_meta(run_id)

        input_path = resolve_dataset_path(dataset_id)
        if dataset_kind == "raw_workload_csv":
            set_stage("loading_model_artifacts")
            inference_output_path = inference_predictions_path(dataset_id)
            inference_batch_path = run_dir / "inference-batch.csv"
            validation = get_dataset_metadata_for_id(dataset_id).get("validation", {})
            validated_vm_ids = validation.get("vm_ids", [])
            expected_inference_vms = (
                len(vm_ids) if vm_ids else len(validated_vm_ids)
            )
            run["inference_total_vm_count"] = expected_inference_vms

            def on_vm_inference_complete(
                completed_count: int,
                completed_vm_id: str | None,
            ) -> None:
                run["inference_completed_vm_count"] = completed_count
                run["inference_last_completed_vm_id"] = completed_vm_id
                _save_meta(run_id)

            def set_inference_stage(stage: str) -> None:
                set_stage(stage)

            prediction_metadata = infer_and_merge_dataset(
                resolve_dataset_path(dataset_id),
                inference_output_path,
                inference_batch_path,
                source_filename,
                vm_ids,
                start_time=start_time,
                end_time=end_time,
                on_stage=set_inference_stage,
                on_vm_complete=on_vm_inference_complete,
            )
            if run.get("inference_completed_vm_count", 0) < expected_inference_vms:
                raise RuntimeError(
                    "Fresh inference did not complete for every selected raw VM: "
                    f"{run.get('inference_completed_vm_count', 0)} of "
                    f"{expected_inference_vms} were processed."
                )
            input_path = inference_output_path
            run["generated_prediction_rows"] = prediction_metadata[
                "checkpoint_row_count"
            ]
            run["generated_prediction_vms"] = prediction_metadata[
                "checkpoint_vm_count"
            ]
            run["inference_processed_vm_count"] = prediction_metadata[
                "processed_vm_count"
            ]
            run["inference_cached_vm_count"] = prediction_metadata[
                "resumed_vm_count"
            ]
            run["inference_generated_rows"] = prediction_metadata[
                "generated_row_count"
            ]
            run["inference_mode"] = (
                "resumed_predictions"
                if prediction_metadata["generated_row_count"] == 0
                else (
                    "fresh_and_resumed_predictions"
                    if prediction_metadata["resumed_vm_count"] > 0
                    else "fresh_predictions"
                )
            )

        selected_input_path = run_dir / "selected_predictions.csv"
        selection_metadata = _prepare_selected_predictions(
            input_path,
            selected_input_path,
            vm_ids,
            start_time,
            end_time,
            risk_threshold,
        )
        run["selected_prediction_rows"] = selection_metadata["row_count"]
        run["selected_prediction_vms"] = selection_metadata["vm_count"]
        set_stage("preparing_isolated_project")
        project_copy, output_dir = _prepare_isolated_project(
            run_id, dataset_id, selected_input_path
        )
        maven = shutil.which("mvn")
        if not maven:
            raise FileNotFoundError("Apache Maven was not found on PATH.")

        args = [
            maven,
            "compile",
            "exec:java",
            f"-Dsim.output.dir={output_dir.resolve()}",
            f"-Dsim.policy={policy}",
        ]
        args.extend(f"-D{key}={value}" for key, value in extra_props.items())

        set_stage("compiling_and_running")
        run["command"] = "Maven compile exec:java (arguments recorded without filesystem paths)"
        _save_meta(run_id)
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(
                args,
                cwd=str(project_copy),
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
            run["pid"] = process.pid
            _save_meta(run_id)
            try:
                exit_code = process.wait(timeout=settings.sim_timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                run["status"] = "TIMEOUT"
                set_stage("timed_out")
                run["error"] = (
                    f"Simulation exceeded the configured {settings.sim_timeout_seconds}-second timeout."
                )
                return

        run["exit_code"] = exit_code
        expected_headers = {
            "simulation_summary.csv": {"metric", "value", "notes"},
            "migration_events.csv": {
                "simulation_time", "source_vm_id", "source_host", "target_host", "event",
            },
            "decision_plan.csv": {
                "timestamp", "vm_id", "decision", "current_host", "proposed_host",
            },
            "host_utilization.csv": {
                "timestamp", "host_id", "current_cpu_fraction", "forecast_cpu_fraction",
                "forecast_ram_mb",
            },
            "final_vm_placement.csv": {
                "source_vm_id", "simulated_vm_id", "final_host", "sample_count",
            },
        }
        output_errors: list[str] = []
        comparison_path = output_dir / "policy_comparison.csv"
        comparison_policies: list[str] = []
        if not comparison_path.is_file():
            output_errors.append("Missing output file 'policy_comparison.csv'.")
        else:
            try:
                with comparison_path.open("r", encoding="utf-8", newline="") as stream:
                    reader = csv.DictReader(stream)
                    if not {"policy", "migration_requests", "successful_migrations"} <= set(
                        reader.fieldnames or []
                    ):
                        output_errors.append(
                            "Output file 'policy_comparison.csv' is missing required columns."
                        )
                    comparison_policies = [
                        str(row.get("policy", "")).upper() for row in reader
                    ]
                if len(comparison_policies) != len(set(comparison_policies)):
                    output_errors.append(
                        "Output file 'policy_comparison.csv' contains duplicate policies."
                    )
                missing_policies = set(SUPPORTED_POLICIES) - set(comparison_policies)
                unexpected_policies = set(comparison_policies) - set(SUPPORTED_POLICIES)
                if missing_policies:
                    output_errors.append(
                        "Output file 'policy_comparison.csv' is missing policies: "
                        + ", ".join(sorted(missing_policies))
                        + "."
                    )
                if unexpected_policies:
                    output_errors.append(
                        "Output file 'policy_comparison.csv' contains unsupported policies: "
                        + ", ".join(sorted(unexpected_policies))
                        + "."
                    )
            except (OSError, UnicodeDecodeError, csv.Error):
                output_errors.append("Output file 'policy_comparison.csv' is unreadable.")
        for scenario_policy in SUPPORTED_POLICIES:
            scenario_dir = settings.run_policy_dir(run_id, scenario_policy)
            for filename, required_headers in expected_headers.items():
                path = scenario_dir / filename
                if not path.is_file():
                    output_errors.append(
                        f"Missing {scenario_policy} output file '{filename}'."
                    )
                    continue
                try:
                    with path.open("r", encoding="utf-8", newline="") as stream:
                        actual_headers = set(next(csv.reader(stream), []))
                except (OSError, UnicodeDecodeError, csv.Error):
                    output_errors.append(
                        f"{scenario_policy} output file '{filename}' is unreadable."
                    )
                    continue
                missing_headers = required_headers - actual_headers
                if missing_headers:
                    output_errors.append(
                        f"{scenario_policy} output file '{filename}' is missing columns: "
                        f"{', '.join(sorted(missing_headers))}."
                    )
            summary_path = scenario_dir / "simulation_summary.csv"
            if summary_path.is_file():
                try:
                    with summary_path.open("r", encoding="utf-8", newline="") as stream:
                        summary_rows = csv.DictReader(stream)
                        if not any(
                            row.get("metric") == "energy_kwh" and row.get("value")
                            for row in summary_rows
                        ):
                            output_errors.append(
                                f"{scenario_policy} summary does not report energy_kwh."
                            )
                except (OSError, UnicodeDecodeError, csv.Error):
                    output_errors.append(
                        f"{scenario_policy} output file 'simulation_summary.csv' is unreadable."
                    )
        if exit_code != 0:
            run["status"] = "FAILED"
            set_stage("failed")
            run["error"] = f"Maven exited with code {exit_code}. See the run logs."
        elif output_errors:
            run["status"] = "FAILED"
            set_stage("failed")
            run["error"] = "Simulator output validation failed: " + " ".join(output_errors)
        else:
            run["status"] = "COMPLETED"
            set_stage("completed")
            run["progress"] = 100
            run["error"] = None
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        logger.exception("Run %s failed during stage %s", run_id, run.get("stage"))
        run["status"] = "FAILED"
        set_stage("failed")
        run["error"] = str(exc)
    finally:
        durations = run.setdefault("stage_durations_seconds", {})
        current_stage = str(run.get("stage", "unknown"))
        durations[current_stage] = round(
            durations.get(current_stage, 0.0) + time.perf_counter() - stage_started,
            3,
        )
        run["finished_at"] = datetime.now(timezone.utc).isoformat()
        run.pop("pid", None)
        _save_meta(run_id)
        with _run_lock:
            if _active_run_id == run_id:
                _active_run_id = None


def submit_run(
    policy: str,
    dataset_id: str | None = None,
    extra_props: dict[str, str] | None = None,
    vm_ids: list[str] | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    risk_threshold: float = 0.66,
) -> dict[str, Any]:
    selected_policy = policy.upper()
    if selected_policy not in SUPPORTED_POLICIES:
        raise ValueError(f"Unsupported policy '{policy}'. Must be one of {SUPPORTED_POLICIES}.")
    if not settings.sim_execution_enabled:
        raise RuntimeError("Simulation execution is disabled (SIM_EXECUTION_ENABLED=false).")
    start = _selection_datetime(start_time, "start_time")
    end = _selection_datetime(end_time, "end_time")
    if start and end and start > end:
        raise ValueError("start_time must be earlier than or equal to end_time.")
    if not 0 <= risk_threshold <= 1:
        raise ValueError("risk_threshold must be within [0, 1].")
    normalized_vm_ids = sorted({str(vm_id).strip() for vm_id in (vm_ids or [])})
    if any(not vm_id for vm_id in normalized_vm_ids):
        raise ValueError("VM IDs must not be empty.")

    selected_dataset_id = dataset_id or get_active_dataset_id()
    input_path = resolve_dataset_path(selected_dataset_id)
    dataset_metadata = get_dataset_metadata_for_id(selected_dataset_id)
    dataset_kind = dataset_metadata.get("dataset_kind", "prediction_csv")
    if dataset_kind == "prediction_csv":
        simulator_errors = simulator_compatibility_errors(input_path)
        if simulator_errors:
            raise SimulatorInputError(
                "Simulator input is incompatible: " + " ".join(simulator_errors)
            )
    elif dataset_kind == "incompatible_raw_workload_csv":
        raise SimulatorInputError(
            "This raw Alibaba dataset is preserved but cannot be simulated: memory "
            "percentage and disk-space percentage do not satisfy the model's memory-KB "
            "and disk-throughput inputs."
        )
    elif dataset_kind != "raw_workload_csv":
        raise SimulatorInputError(
            "Selected dataset is not compatible raw workload or prepared prediction data."
        )

    global _active_run_id
    with _run_lock:
        if _active_run_id is not None:
            active = _runs.get(_active_run_id, {})
            raise RuntimeError(
                f"A simulation is already running (run_id={_active_run_id}, "
                f"policy={active.get('policy')}). Wait for it to finish."
            )
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        _active_run_id = run_id

    run = {
        "run_id": run_id,
        "policy": selected_policy,
        "dataset_id": selected_dataset_id,
        "data_mode": dataset_kind,
        "source_filename": dataset_metadata.get("filename", input_path.name),
        "status": "PENDING",
        "stage": "queued",
        "progress": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "finished_at": None,
        "exit_code": None,
        "error": None,
        "command": None,
        "extra_props": extra_props or {},
        "run_parameters": {
            "vm_ids": normalized_vm_ids or None,
            "start_time": start_time,
            "end_time": end_time,
            "risk_threshold": risk_threshold,
            "host_count": int((extra_props or {}).get("sim.host.count", "200")),
            "host_cpu_limit": float((extra_props or {}).get("sim.host.cpu.limit", "0.80")),
            "host_ram_limit": float((extra_props or {}).get("sim.host.ram.limit", "0.80")),
            "host_ram_mb": int((extra_props or {}).get("sim.host.ram.mb", "262144")),
            "host_pes": int((extra_props or {}).get("sim.host.pes", "32")),
            "compared_policies": SUPPORTED_POLICIES,
        },
        "source": "dashboard",
    }
    _runs[run_id] = run
    _save_meta(run_id)
    thread = threading.Thread(
        target=_run_simulation_thread,
        args=(
            run_id,
            selected_policy,
            extra_props or {},
            selected_dataset_id,
            dataset_kind,
            dataset_metadata.get("filename", input_path.name),
            normalized_vm_ids or None,
            start_time,
            end_time,
            risk_threshold,
        ),
        daemon=True,
        name=f"sim-{run_id}",
    )
    thread.start()
    return run


_load_all_runs()
