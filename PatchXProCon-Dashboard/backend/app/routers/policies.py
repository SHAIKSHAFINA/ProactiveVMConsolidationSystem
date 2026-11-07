"""Saved and freshly simulated policy outputs."""
from __future__ import annotations

import io
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..config import SUPPORTED_POLICIES, settings
from ..data import (
    decision_plan_df,
    final_placement_df,
    host_utilization_df,
    migration_events_df,
    policy_comparison_df,
    read_csv_safe,
    summary_to_dict,
)

router = APIRouter()


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
    run_id: str | None = None,
    dataset_id: str | None = None,
) -> Path:
    selected = _validate_policy(policy)
    if not run_id:
        if dataset_id and dataset_id != "bitbrains":
            raise HTTPException(
                status_code=409,
                detail="No matching simulation results yet for the selected dataset.",
            )
        return settings.policy_dir(selected)

    from ..runner import get_run

    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' was not found.")
    if run.get("status") != "COMPLETED":
        raise HTTPException(
            status_code=409,
            detail=f"Run '{run_id}' has no completed output (status: {run.get('status')}).",
        )
    if dataset_id and run.get("dataset_id") != dataset_id:
        raise HTTPException(
            status_code=409,
            detail="The selected run belongs to a different prediction dataset.",
        )
    candidate = settings.run_policy_dir(run_id, selected)
    if not candidate.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Policy '{selected}' has no output in run '{run_id}'.",
        )
    return candidate


def _reported_dataset_id(run_id: str | None, dataset_id: str | None) -> str:
    if run_id:
        from ..runner import get_run

        run = get_run(run_id)
        if run is not None and isinstance(run.get("dataset_id"), str):
            return str(run["dataset_id"])
    return dataset_id or "bitbrains"


def _safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


@lru_cache(maxsize=16)
def _fresh_comparison_rows(output_root: Path, run_id: str) -> list[dict[str, Any]]:
    frame = _read_result(output_root / "policy_comparison.csv", policy_comparison_df)
    _require_columns(
        frame,
        {
            "policy",
            "migration_requests",
            "successful_migrations",
            "failed_or_incomplete_migrations",
            "no_feasible_destination_decisions",
            "decision_rows",
        },
        "policy_comparison.csv",
    )
    frame["policy"] = frame["policy"].astype(str).str.upper()
    if frame["policy"].duplicated().any():
        raise HTTPException(
            status_code=422,
            detail="policy_comparison.csv contains duplicate policy rows.",
        )
    unknown = set(frame["policy"]) - set(SUPPORTED_POLICIES)
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"policy_comparison.csv contains unsupported policies: {sorted(unknown)}.",
        )

    rows: list[dict[str, Any]] = []
    count_columns = (
        "migration_requests",
        "successful_migrations",
        "failed_or_incomplete_migrations",
        "no_feasible_destination_decisions",
        "decision_rows",
    )
    for source_row in _safe_records(frame):
        policy = str(source_row["policy"])
        policy_dir = settings.run_policy_dir(run_id, policy)
        summary = _read_result(policy_dir / "simulation_summary.csv", summary_to_dict)
        host_frame = _read_result(
            policy_dir / "host_utilization.csv",
            host_utilization_df,
        )
        _require_columns(
            host_frame,
            {
                "timestamp",
                "host_id",
                "forecast_cpu_fraction",
                "forecast_ram_mb",
            },
            f"{policy}/host_utilization.csv",
        )
        row: dict[str, Any] = {"policy": policy, "run_id": run_id}
        for column in count_columns:
            value = pd.to_numeric(
                pd.Series([source_row.get(column)]),
                errors="coerce",
            ).iloc[0]
            row[column] = int(value) if pd.notna(value) else None
        energy = pd.to_numeric(
            pd.Series([summary.get("energy_kwh")]),
            errors="coerce",
        ).iloc[0]
        row["energy_kwh"] = (
            float(energy) if pd.notna(energy) and math.isfinite(float(energy)) else None
        )

        try:
            cpu_limit = float(summary["cpu_limit_fraction"])
            ram_limit_mb = float(summary["host_ram_mb"]) * float(
                summary["ram_limit_fraction"]
            )
            cpu = pd.to_numeric(host_frame["forecast_cpu_fraction"], errors="coerce")
            ram = pd.to_numeric(host_frame["forecast_ram_mb"], errors="coerce")
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"{policy} summary is missing valid overload thresholds.",
            ) from exc
        overloaded = host_frame.loc[
            (cpu > cpu_limit) | (ram > ram_limit_mb),
            ["timestamp", "host_id"],
        ].drop_duplicates()
        row["host_overload_snapshots"] = int(len(overloaded))
        for summary_metric in ("host_cpu_overload_snapshots", "host_ram_overload_snapshots"):
            metric = pd.to_numeric(
                pd.Series([summary.get(summary_metric)]),
                errors="coerce",
            ).iloc[0]
            row[summary_metric] = int(metric) if pd.notna(metric) else None
        rows.append(row)
    return rows


def _require_columns(frame: pd.DataFrame, required: set[str], filename: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"{filename} is missing required columns: {', '.join(sorted(missing))}.",
        )


def _read_result(path: Path, reader: Any) -> pd.DataFrame:
    try:
        return reader(path)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Required result CSV '{path.name}' is unavailable.",
        ) from exc
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Unable to read result CSV '{path.name}': {type(exc).__name__}.",
        ) from exc


@router.get("/overview")
def get_overview(
    run_id: str | None = Query(default=None, max_length=64),
    dataset_id: str | None = Query(default=None, max_length=32),
) -> dict[str, Any]:
    selected_policy = "ML_PROACTIVE"
    if run_id:
        from ..runner import get_run

        run = get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' was not found.")
        if run.get("status") != "COMPLETED":
            raise HTTPException(
                status_code=409,
                detail=f"Run '{run_id}' has no completed output (status: {run.get('status')}).",
            )
        if dataset_id and run.get("dataset_id") != dataset_id:
            raise HTTPException(
                status_code=409,
                detail="The selected run belongs to a different prediction dataset.",
            )
        selected_policy = _validate_policy(str(run["policy"]))
        output_root = settings.run_output_dir(run_id)
        summary_path = _result_dir(selected_policy, run_id) / "simulation_summary.csv"
        data_source = "fresh_run"
    else:
        if dataset_id and dataset_id != "bitbrains":
            raise HTTPException(
                status_code=409,
                detail="No matching simulation results yet for the selected dataset.",
            )
        comparison_path = settings.sim_saved_results_dir / "policy_comparison.csv"
        summary_path = settings.policy_dir(selected_policy) / "simulation_summary.csv"
        data_source = "saved_results_v10"
    summary = _read_result(summary_path, summary_to_dict)
    comparison = (
        _fresh_comparison_rows(settings.run_output_dir(run_id), run_id)
        if run_id
        else _safe_records(_read_result(comparison_path, policy_comparison_df))
    )
    return {
        "data_source": data_source,
        "run_id": run_id,
        "dataset_id": run.get("dataset_id") if run_id else "bitbrains",
        "primary_policy": selected_policy,
        "primary_summary": summary,
        "policy_comparison": comparison,
        "supported_policies": SUPPORTED_POLICIES,
        "metric_definitions": {
            "energy_kwh": "Approximate simulated energy estimate (kWh), not a measured facility meter.",
            "host_overload_snapshots": (
                "Unique host/timestamp snapshots whose forecast CPU or RAM exceeds "
                "the simulator threshold; a simulation proxy, not an SLA violation."
            ),
            "host_capacity": "Host CPU/RAM capacities are explicit simulation assumptions.",
            "disk_io_constraints": "NOT_MODELED by the simulator.",
        },
    }


@router.get("/policies")
def list_policies() -> dict[str, Any]:
    comparison_path = settings.sim_saved_results_dir / "policy_comparison.csv"
    comparison = _read_result(comparison_path, policy_comparison_df)
    return {
        "policies": SUPPORTED_POLICIES,
        "comparison": _safe_records(comparison),
        "data_source": "saved_results_v10",
    }


@router.get("/policies/{policy}/summary")
def get_policy_summary(
    policy: str,
    run_id: str | None = Query(default=None, max_length=64),
    dataset_id: str | None = Query(default=None, max_length=32),
) -> dict[str, Any]:
    selected = _validate_policy(policy)
    result_dir = _result_dir(selected, run_id, dataset_id)
    summary = _read_result(result_dir / "simulation_summary.csv", summary_to_dict)
    return {
        "policy": selected,
        "run_id": run_id,
        "dataset_id": _reported_dataset_id(run_id, dataset_id),
        "data_source": "fresh_run" if run_id else "saved_results_v10",
        "summary": summary,
    }


@router.get("/policies/{policy}/migrations")
def get_policy_migrations(
    policy: str,
    run_id: str | None = Query(default=None, max_length=64),
    dataset_id: str | None = Query(default=None, max_length=32),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=1, le=1000),
    event_filter: str | None = Query(default=None, max_length=30),
    vm_filter: str | None = Query(default=None, max_length=100),
    host_filter: str | None = Query(default=None, max_length=64),
    sort_by: str = Query(default="simulation_time", max_length=40),
    sort_order: str = Query(default="asc", pattern="^(?i:asc|desc)$"),
) -> dict[str, Any]:
    selected = _validate_policy(policy)
    frame = _read_result(
        _result_dir(selected, run_id, dataset_id) / "migration_events.csv",
        migration_events_df,
    )
    _require_columns(
        frame,
        {"simulation_time", "source_vm_id", "source_host", "target_host", "event", "details"},
        "migration_events.csv",
    )
    if event_filter:
        frame = frame[frame["event"].str.upper() == event_filter.upper()]
    if host_filter:
        frame = frame[
            (frame["source_host"].astype(str) == host_filter)
            | (frame["target_host"].astype(str) == host_filter)
        ]
    if vm_filter:
        matches = frame.astype(str).apply(
            lambda column: column.str.contains(vm_filter, case=False, na=False, regex=False)
        ).any(axis=1)
        frame = frame[matches]
    if sort_by not in {"simulation_time", "source_vm_id", "source_host", "target_host", "event", "details"}:
        raise HTTPException(status_code=400, detail="Unsupported migration sort column.")
    if sort_by == "simulation_time":
        frame["_sort_value"] = pd.to_numeric(frame[sort_by], errors="coerce")
        frame = frame.sort_values(
            "_sort_value",
            ascending=sort_order.lower() == "asc",
            kind="mergesort",
            na_position="last",
        ).drop(columns="_sort_value")
    else:
        frame = frame.sort_values(
            sort_by,
            ascending=sort_order.lower() == "asc",
            kind="mergesort",
            key=lambda values: values.astype(str).str.lower(),
        )
    total = len(frame)
    start = (page - 1) * page_size
    return {
        "policy": selected,
        "run_id": run_id,
        "dataset_id": _reported_dataset_id(run_id, dataset_id),
        "total": total,
        "page": page,
        "page_size": page_size,
        "sort_by": sort_by,
        "sort_order": sort_order.lower(),
        "records": _safe_records(frame.iloc[start : start + page_size]),
    }


@router.get("/policies/{policy}/decisions")
def get_policy_decisions(
    policy: str,
    run_id: str | None = Query(default=None, max_length=64),
    dataset_id: str | None = Query(default=None, max_length=32),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=1, le=1000),
    decision_filter: str | None = Query(default=None, max_length=30),
    vm_filter: str | None = Query(default=None, max_length=100),
    host_filter: str | None = Query(default=None, max_length=64),
    sort_by: str = Query(default="timestamp", max_length=40),
    sort_order: str = Query(default="asc", pattern="^(?i:asc|desc)$"),
) -> dict[str, Any]:
    selected = _validate_policy(policy)
    frame = _read_result(
        _result_dir(selected, run_id, dataset_id) / "decision_plan.csv",
        decision_plan_df,
    )
    _require_columns(
        frame,
        {
            "timestamp", "vm_id", "risk_probability", "risk_threshold",
            "decision", "current_host", "proposed_host", "reason",
        },
        "decision_plan.csv",
    )
    if decision_filter:
        if decision_filter.upper() == "ACTIONABLE":
            frame = frame[frame["decision"].str.upper() != "KEEP"]
        else:
            frame = frame[frame["decision"].str.upper() == decision_filter.upper()]
    if host_filter:
        frame = frame[
            (frame["current_host"].astype(str) == host_filter)
            | (frame["proposed_host"].astype(str) == host_filter)
        ]
    if vm_filter:
        matches = frame.astype(str).apply(
            lambda column: column.str.contains(vm_filter, case=False, na=False, regex=False)
        ).any(axis=1)
        frame = frame[matches]
    if sort_by not in {
        "timestamp", "vm_id", "risk_probability", "risk_threshold",
        "decision", "current_host", "proposed_host", "reason",
    }:
        raise HTTPException(status_code=400, detail="Unsupported decision sort column.")
    if sort_by in {"risk_probability", "risk_threshold"}:
        frame["_sort_value"] = pd.to_numeric(frame[sort_by], errors="coerce")
        frame = frame.sort_values(
            "_sort_value",
            ascending=sort_order.lower() == "asc",
            kind="mergesort",
            na_position="last",
        ).drop(columns="_sort_value")
    else:
        frame = frame.sort_values(
            sort_by,
            ascending=sort_order.lower() == "asc",
            kind="mergesort",
            key=lambda values: values.astype(str).str.lower(),
        )
    total = len(frame)
    start = (page - 1) * page_size
    return {
        "policy": selected,
        "run_id": run_id,
        "dataset_id": _reported_dataset_id(run_id, dataset_id),
        "total": total,
        "page": page,
        "page_size": page_size,
        "sort_by": sort_by,
        "sort_order": sort_order.lower(),
        "records": _safe_records(frame.iloc[start : start + page_size]),
    }


@router.get("/policies/{policy}/hosts")
def get_policy_hosts(
    policy: str,
    run_id: str | None = Query(default=None, max_length=64),
    dataset_id: str | None = Query(default=None, max_length=32),
    host_id: str | None = Query(default=None, max_length=64),
) -> dict[str, Any]:
    selected = _validate_policy(policy)
    frame = _read_result(
        _result_dir(selected, run_id, dataset_id) / "host_utilization.csv",
        host_utilization_df,
    )
    required = {"timestamp", "host_id", "current_cpu_fraction", "forecast_cpu_fraction",
                "current_ram_mb", "forecast_ram_mb", "current_ram_fraction",
                "forecast_ram_fraction", "vm_count"}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise HTTPException(
            status_code=422,
            detail=f"host_utilization.csv is missing columns: {sorted(missing_columns)}",
        )
    numeric_columns = [
        "current_cpu_fraction",
        "forecast_cpu_fraction",
        "current_ram_mb",
        "forecast_ram_mb",
        "current_ram_fraction",
        "forecast_ram_fraction",
        "vm_count",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if host_id is not None:
        frame = frame[frame["host_id"].astype(str) == host_id]
    frame = frame.sort_values("timestamp")
    if frame.empty:
        return {
            "policy": selected,
            "dataset_id": _reported_dataset_id(run_id, dataset_id),
            "run_id": run_id,
            "host_count": 0,
            "hosts": [],
            "snapshots": [],
        }
    latest = frame.groupby("host_id", sort=True).tail(1).copy()
    latest["snapshot_count"] = latest["host_id"].map(frame.groupby("host_id").size())
    latest = latest.rename(
        columns={
            "current_cpu_fraction": "current_cpu_fraction",
            "forecast_cpu_fraction": "forecast_cpu_fraction",
            "current_ram_fraction": "current_ram_fraction",
            "forecast_ram_fraction": "forecast_ram_fraction",
            "vm_count": "vm_count_at_snapshot",
        }
    )
    return {
        "policy": selected,
        "run_id": run_id,
        "dataset_id": _reported_dataset_id(run_id, dataset_id),
        "host_count": int(frame["host_id"].nunique()),
        "hosts": _safe_records(latest),
        "snapshots": _safe_records(frame.tail(1000)),
    }


@router.get("/policies/{policy}/placements")
def get_policy_placements(
    policy: str,
    run_id: str | None = Query(default=None, max_length=64),
    dataset_id: str | None = Query(default=None, max_length=32),
    vm_filter: str | None = Query(default=None, max_length=100),
    host_filter: str | None = Query(default=None, max_length=64),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=200, ge=1, le=2000),
) -> dict[str, Any]:
    selected = _validate_policy(policy)
    frame = _read_result(
        _result_dir(selected, run_id, dataset_id) / "final_vm_placement.csv",
        final_placement_df,
    )
    _require_columns(frame, {"source_vm_id", "final_host"}, "final_vm_placement.csv")
    if vm_filter:
        frame = frame[
            frame["source_vm_id"].str.contains(vm_filter, case=False, na=False, regex=False)
        ]
    if host_filter:
        frame = frame[frame["final_host"].astype(str) == host_filter]
    total = len(frame)
    start = (page - 1) * page_size
    return {
        "policy": selected,
        "run_id": run_id,
        "dataset_id": _reported_dataset_id(run_id, dataset_id),
        "total": total,
        "page": page,
        "page_size": page_size,
        "records": _safe_records(frame.iloc[start : start + page_size]),
    }


@router.get("/export/policy_comparison")
def export_policy_comparison(
    dataset_id: str | None = Query(default=None, max_length=32),
    source: str = Query(default="saved", pattern="^(saved|fresh)$"),
    run_id: str | None = Query(default=None, max_length=64),
    search: str | None = Query(default=None, max_length=100),
    sort_by: str = Query(default="energy_kwh", max_length=50),
    sort_order: str = Query(default="asc", pattern="^(?i:asc|desc)$"),
) -> StreamingResponse:
    if source == "fresh":
        if not dataset_id:
            raise HTTPException(
                status_code=400,
                detail="dataset_id is required when exporting fresh run results.",
            )
        from ..runner import list_runs

        matching_runs = [
            run for run in list_runs()
            if run.get("dataset_id") == dataset_id and run.get("status") == "COMPLETED"
        ]
        if matching_runs:
            selected_run = next(
                (run for run in matching_runs if run.get("run_id") == run_id),
                None,
            ) if run_id else matching_runs[0]
            if selected_run is None:
                raise HTTPException(
                    status_code=409,
                    detail="The selected run is not a completed run for this dataset.",
                )
            rows = _fresh_comparison_rows(
                settings.run_output_dir(str(selected_run["run_id"])),
                str(selected_run["run_id"]),
            )
            frame = pd.DataFrame(rows)
            if search:
                frame = frame[
                    frame["policy"].str.contains(search, case=False, regex=False, na=False)
                ]
            sortable_columns = {
                "energy_kwh",
                "migration_requests",
                "successful_migrations",
                "failed_or_incomplete_migrations",
                "host_overload_snapshots",
            }
            if sort_by not in sortable_columns:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported policy comparison sort column.",
                )
            frame["_sort_value"] = pd.to_numeric(frame[sort_by], errors="coerce")
            frame = frame.sort_values(
                ["_sort_value", "policy"],
                ascending=[sort_order.lower() == "asc", True],
                kind="mergesort",
                na_position="last",
            ).drop(columns="_sort_value")
            content = frame.to_csv(index=False)
        if not matching_runs:
            raise HTTPException(
                status_code=409,
                detail="No matching simulation results yet for the selected dataset.",
            )
    else:
        if dataset_id and dataset_id != "bitbrains":
            raise HTTPException(
                status_code=409,
                detail="Saved v10 baseline results are only available for the Bitbrains dataset.",
            )
        path = settings.sim_saved_results_dir / "policy_comparison.csv"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Saved policy comparison is unavailable.")
        content = path.read_text(encoding="utf-8")
    return StreamingResponse(
        io.StringIO(content),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=policy_comparison.csv"},
    )


@router.get("/export/{policy}/migrations")
def export_migrations(
    policy: str,
    run_id: str | None = Query(default=None, max_length=64),
    dataset_id: str | None = Query(default=None, max_length=32),
) -> StreamingResponse:
    selected = _validate_policy(policy)
    frame = _read_result(
        _result_dir(selected, run_id, dataset_id) / "migration_events.csv",
        migration_events_df,
    )
    content = frame.to_csv(index=False)
    filename = f"{selected.lower()}_migration_events.csv"
    return StreamingResponse(
        io.StringIO(content),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/export/{policy}/decisions")
def export_decisions(
    policy: str,
    run_id: str | None = Query(default=None, max_length=64),
    dataset_id: str | None = Query(default=None, max_length=32),
) -> StreamingResponse:
    selected = _validate_policy(policy)
    frame = _read_result(
        _result_dir(selected, run_id, dataset_id) / "decision_plan.csv",
        decision_plan_df,
    )
    content = frame.to_csv(index=False)
    filename = f"{selected.lower()}_decision_plan.csv"
    return StreamingResponse(
        io.StringIO(content),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
