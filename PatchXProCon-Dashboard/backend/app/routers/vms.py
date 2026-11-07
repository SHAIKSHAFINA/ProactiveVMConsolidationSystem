"""VM search and details sourced from the selected prediction CSV and result files."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from ..config import SUPPORTED_POLICIES, settings
from ..data import decision_plan_df, final_placement_df, migration_events_df
from .dataset import (
    get_active_dataset_id,
    get_dataset_metadata_for_id,
    inference_predictions_path,
    read_prediction_frame,
)

router = APIRouter()


def _safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


def _validate_policy(policy: str) -> str:
    selected = policy.upper()
    if selected not in SUPPORTED_POLICIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown policy '{policy}'. Supported: {SUPPORTED_POLICIES}",
        )
    return selected


def _result_dir(
    policy: str,
    dataset_id: str,
    run_id: str | None,
) -> Path | None:
    if run_id:
        from ..runner import get_run

        run = get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' was not found.")
        if run.get("status") != "COMPLETED":
            raise HTTPException(
                status_code=409,
                detail=f"Run '{run_id}' has no completed output.",
            )
        if run.get("dataset_id") != dataset_id:
            raise HTTPException(
                status_code=409,
                detail="The selected run belongs to a different prediction dataset.",
            )
        return settings.run_policy_dir(run_id, policy)
    if dataset_id == "bitbrains":
        return settings.policy_dir(policy)
    return None


@router.get("/vms")
def list_vms(
    search: str | None = Query(default=None, max_length=100),
    dataset_id: str | None = Query(default=None, max_length=32),
    run_id: str | None = Query(default=None, max_length=64),
    policy: str = Query(default="ML_PROACTIVE"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=1, le=2000),
) -> dict[str, Any]:
    selected_policy = _validate_policy(policy)
    selected_dataset_id = dataset_id or get_active_dataset_id()
    try:
        dataset_df = read_prediction_frame(selected_dataset_id, run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise HTTPException(status_code=422, detail=f"Unable to read selected dataset: {exc}") from exc

    vm_ids = sorted(dataset_df["vm_id"].dropna().astype(str).unique())
    rows = pd.DataFrame({"source_vm_id": vm_ids})
    result_dir = _result_dir(selected_policy, selected_dataset_id, run_id)
    placement_path = result_dir / "final_vm_placement.csv" if result_dir else None
    placements = (
        final_placement_df(placement_path)
        if placement_path is not None and placement_path.is_file()
        else pd.DataFrame()
    )
    if not placements.empty and {"source_vm_id", "final_host"}.issubset(placements.columns):
        placement_columns = [
            column
            for column in (
                "source_vm_id",
                "final_host",
                "provisioned_ram_mb",
                "sample_count",
                "risk_alert_samples",
                "clamped_forecast_rows",
                "sla_overload_intervals",
            )
            if column in placements.columns
        ]
        rows = rows.merge(
            placements[placement_columns].drop_duplicates("source_vm_id"),
            on="source_vm_id",
            how="left",
        )
    if search:
        rows = rows[
            rows["source_vm_id"].str.contains(search, case=False, na=False, regex=False)
        ]

    total = len(rows)
    start = (page - 1) * page_size
    page_rows = rows.iloc[start : start + page_size]
    return {
        "policy": selected_policy,
        "dataset_id": selected_dataset_id,
        "run_id": run_id,
        "total": total,
        "page": page,
        "page_size": page_size,
        "vms": _safe_records(page_rows),
    }


@router.get("/vms/{vm_id}")
def get_vm_detail(
    vm_id: str,
    dataset_id: str | None = Query(default=None, max_length=32),
    run_id: str | None = Query(default=None, max_length=64),
    policy: str = Query(default="ML_PROACTIVE"),
    start_time: str | None = Query(default=None, max_length=40),
    end_time: str | None = Query(default=None, max_length=40),
) -> dict[str, Any]:
    selected_policy = _validate_policy(policy)
    selected_dataset_id = dataset_id or get_active_dataset_id()
    try:
        input_df = read_prediction_frame(selected_dataset_id, run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise HTTPException(status_code=422, detail=f"Unable to read selected dataset: {exc}") from exc

    vm_rows = input_df[input_df["vm_id"].astype(str) == str(vm_id)].copy()
    if vm_rows.empty:
        raise HTTPException(status_code=404, detail=f"VM '{vm_id}' was not found in the selected dataset.")
    try:
        start = pd.Timestamp(start_time) if start_time else None
        end = pd.Timestamp(end_time) if end_time else None
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Prediction time bounds must be ISO local datetimes.",
        ) from exc
    if start is not None and start.tzinfo is not None or end is not None and end.tzinfo is not None:
        raise HTTPException(status_code=400, detail="Prediction time bounds must not include a timezone.")
    if start is not None and end is not None and start > end:
        raise HTTPException(status_code=400, detail="start_time must be earlier than or equal to end_time.")

    result_dir = _result_dir(selected_policy, selected_dataset_id, run_id)
    placement: dict[str, Any] | None = None
    placement_path = result_dir / "final_vm_placement.csv" if result_dir else None
    if placement_path is not None and placement_path.is_file():
        placement_df = final_placement_df(placement_path)
        matching_placements = placement_df[
            placement_df["source_vm_id"].astype(str) == str(vm_id)
        ]
        if not matching_placements.empty:
            placement = _safe_records(matching_placements.head(1))[0]

    migration_path = result_dir / "migration_events.csv" if result_dir else None
    migrations: list[dict[str, Any]] = []
    if migration_path is not None and migration_path.is_file():
        migration_df = migration_events_df(migration_path)
        migrations = _safe_records(
            migration_df[migration_df["source_vm_id"].astype(str) == str(vm_id)]
        )

    decision_path = result_dir / "decision_plan.csv" if result_dir else None
    decisions: list[dict[str, Any]] = []
    if decision_path is not None and decision_path.is_file():
        decision_df = decision_plan_df(decision_path)
        matching_decisions = decision_df[
            decision_df["vm_id"].astype(str) == str(vm_id)
        ].copy()
        if not matching_decisions.empty:
            matching_decisions["_parsed_time"] = pd.to_datetime(
                matching_decisions["timestamp"], errors="coerce"
            )
            if start is not None:
                matching_decisions = matching_decisions[
                    matching_decisions["_parsed_time"] >= start
                ]
            if end is not None:
                matching_decisions = matching_decisions[
                    matching_decisions["_parsed_time"] <= end
                ]
            matching_decisions = matching_decisions.sort_values(
                ["_parsed_time", "timestamp"],
                kind="mergesort",
                na_position="last",
            ).drop(columns="_parsed_time")
            decisions = _safe_records(matching_decisions)

    preferred_columns = [
        "decision_time",
        "current_cpu_pct",
        "forecast_cpu_max",
        "current_memory_kb",
        "forecast_memory_max",
        "risk_probability",
        "risk_threshold",
        "risk_alert",
        "current_disk_read_kbps",
        "current_disk_write_kbps",
        "forecast_disk_read_max",
        "forecast_disk_write_max",
    ]
    vm_rows = vm_rows.assign(
        _parsed_time=pd.to_datetime(vm_rows["decision_time"], errors="coerce")
    )
    if start is not None:
        vm_rows = vm_rows[vm_rows["_parsed_time"] >= start]
    if end is not None:
        vm_rows = vm_rows[vm_rows["_parsed_time"] <= end]
    vm_rows = vm_rows.sort_values("_parsed_time").drop(columns="_parsed_time")
    vm_rows["simulator_forecast_cpu_pct"] = (
        pd.to_numeric(vm_rows["forecast_cpu_max"], errors="coerce")
        .clip(lower=0, upper=100)
    )
    preferred_columns.append("simulator_forecast_cpu_pct")
    prediction_columns = [column for column in preferred_columns if column in vm_rows.columns]
    series = _safe_records(vm_rows[prediction_columns])
    disk_available = any(
        column in vm_rows.columns
        for column in (
            "current_disk_read_kbps",
            "current_disk_write_kbps",
            "forecast_disk_read_max",
            "forecast_disk_write_max",
        )
    )
    data_mode = "prepared_prediction_csv"
    if run_id:
        from ..runner import get_run

        selected_run = get_run(run_id)
        if selected_run is not None and selected_run.get("data_mode") == "raw_workload_csv":
            data_mode = "fresh_model_inference"
    elif (
        get_dataset_metadata_for_id(selected_dataset_id).get("dataset_kind")
        == "raw_workload_csv"
        and inference_predictions_path(selected_dataset_id).is_file()
    ):
        data_mode = "fresh_model_inference"

    return {
        "vm_id": str(vm_id),
        "dataset_id": selected_dataset_id,
        "run_id": run_id,
        "policy": selected_policy,
        "placement": placement,
        "migrations": migrations,
        "decisions": decisions,
        "prediction_series": series,
        "disk_forecasts_available": disk_available,
        "prediction_data_mode": data_mode,
        "prediction_series_note": (
            "CPU, memory, disk, and risk values were generated by the configured PatchTST "
            "and Logistic Regression artifacts for this dataset."
            if data_mode == "fresh_model_inference"
            else "CPU, memory, disk, and risk values are recorded values from the prepared "
            "prediction CSV; no fresh inference was performed."
        )
        + " simulator_forecast_cpu_pct is the simulator's documented forecast_cpu_max "
        "clamp to [0, 100].",
        "disk_constraint_note": "NOT_MODELED: simulator does not enforce host disk I/O capacity.",
        "placement_note": (
            (
                f"Placement and migration history are from run {run_id}."
                if run_id
                else "Placement and migration history are from saved v10 results for the Bitbrains demo."
                if selected_dataset_id == "bitbrains"
                else "No matching simulation results yet for this dataset; no unrelated placement or migration data is shown."
            )
        ),
    }
