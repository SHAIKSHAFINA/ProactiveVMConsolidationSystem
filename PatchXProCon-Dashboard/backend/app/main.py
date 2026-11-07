"""
PatchXProCon Dashboard — FastAPI backend entry point.
"""
from __future__ import annotations

import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .inference import model_inference_status
from .routers import policies, vms, runs, dataset

app = FastAPI(
    title="PatchXProCon Dashboard API",
    description=(
        "Research demonstration API for PatchXProCon — "
        "Patch-Based Predictive and Migration-Aware VM Consolidation."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# CORS — allow the local Vite dev server and production build
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(policies.router, prefix="/api", tags=["Policies & Results"])
app.include_router(vms.router, prefix="/api", tags=["VMs"])
app.include_router(runs.router, prefix="/api", tags=["Simulation Runs"])
app.include_router(dataset.router, prefix="/api", tags=["Dataset"])


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/api/health", tags=["Health"])
def health() -> dict:
    saved_results_ok = settings.sim_saved_results_dir.exists()
    input_csv_ok = settings.sim_input_csv.exists()
    sim_project_ok = settings.sim_project_dir.exists()
    inference = model_inference_status()
    inference_missing = inference["missing"]

    return {
        "status": "ok",
        "version": "1.0.0",
        "saved_results_available": saved_results_ok,
        "input_csv_available": input_csv_ok,
        "sim_project_available": sim_project_ok,
        "sim_execution_enabled": settings.sim_execution_enabled,
        "model_inference_available": inference["available"],
        "model_artifacts": inference["artifacts"],
        "model_configuration": {
            "feature_order": inference["feature_order"],
            "lookback_steps": inference["lookback_steps"],
            "forecast_horizon_steps": inference["forecast_horizon_steps"],
            "classifier_feature_count": inference["classifier_feature_count"],
            "risk_threshold": inference["risk_threshold"],
        },
        "model_inference_missing": inference_missing,
        "model_inference_note": (
            "Live inference is unavailable: " + " ".join(inference_missing)
            if inference_missing
            else "Local PatchTST and Logistic Regression inference artifacts are configured."
        ),
        "python_version": sys.version,
        "scikit_learn_version": inference["sklearn_version"],
        "torch_version": inference["torch_version"],
    }
