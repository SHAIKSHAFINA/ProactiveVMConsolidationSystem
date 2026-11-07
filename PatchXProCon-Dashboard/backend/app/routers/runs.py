"""
Simulation run management endpoints.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

from ..config import SUPPORTED_POLICIES
from ..runner import (
    SimulatorInputError,
    get_run,
    get_run_logs,
    list_runs,
    submit_run,
)

router = APIRouter()


class RunRequest(BaseModel):
    policy: str
    dataset_id: Optional[str] = None
    cpu_limit: Optional[float] = Field(default=None, ge=0.1, le=1.0)
    ram_limit: Optional[float] = Field(default=None, ge=0.1, le=1.0)
    host_count: Optional[int] = Field(default=None, ge=10, le=500)
    host_ram_mb: Optional[int] = Field(default=None, ge=1024, le=2_097_152)
    host_pes: Optional[int] = Field(default=None, ge=4, le=256)
    risk_threshold: float = Field(default=0.66, ge=0.0, le=1.0)
    vm_ids: Optional[list[str]] = Field(default=None, max_length=100)
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    @field_validator("policy")
    @classmethod
    def validate_policy(cls, v: str) -> str:
        p = v.upper()
        if p not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Unsupported policy '{v}'. Must be one of {SUPPORTED_POLICIES}."
            )
        return p

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
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{info.field_name} must be an ISO local datetime.") from exc
        if parsed.tzinfo is not None:
            raise ValueError(f"{info.field_name} must not include a timezone.")
        return value

    @model_validator(mode="after")
    def validate_time_window(self) -> "RunRequest":
        if self.start_time and self.end_time:
            if datetime.fromisoformat(self.start_time) > datetime.fromisoformat(self.end_time):
                raise ValueError("start_time must be earlier than or equal to end_time.")
        return self


@router.get("/runs")
def get_runs() -> dict[str, Any]:
    return {"runs": list_runs()}


@router.post("/runs", status_code=202)
def create_run(req: RunRequest) -> dict[str, Any]:
    extra: dict[str, str] = {}
    if req.cpu_limit is not None:
        extra["sim.host.cpu.limit"] = str(req.cpu_limit)
    if req.ram_limit is not None:
        extra["sim.host.ram.limit"] = str(req.ram_limit)
    if req.host_count is not None:
        extra["sim.host.count"] = str(req.host_count)
    if req.host_ram_mb is not None:
        extra["sim.host.ram.mb"] = str(req.host_ram_mb)
    if req.host_pes is not None:
        extra["sim.host.pes"] = str(req.host_pes)

    try:
        run = submit_run(
            req.policy,
            req.dataset_id,
            extra,
            req.vm_ids,
            req.start_time,
            req.end_time,
            req.risk_threshold,
        )
        return {"run": run}
    except SimulatorInputError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/runs/{run_id}")
def get_run_status(run_id: str) -> dict[str, Any]:
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return {"run": run}


@router.get("/runs/{run_id}/logs")
def get_run_log(run_id: str) -> dict[str, Any]:
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    logs = get_run_logs(run_id)
    return {"run_id": run_id, **logs}
