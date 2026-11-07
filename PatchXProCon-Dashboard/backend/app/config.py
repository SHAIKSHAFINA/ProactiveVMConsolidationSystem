"""
Configuration — reads from .env in the backend directory.
All filesystem paths are resolved and validated at startup.
"""
from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Simulator (read-only)
    sim_project_dir: Path = PROJECT_ROOT / "PatchTST-VM-Consolidation"
    sim_saved_results_dir: Path = (
        PROJECT_ROOT / "PatchTST-VM-Consolidation" / "results_v10"
    )
    sim_input_csv: Path = (
        PROJECT_ROOT
        / "PatchTST-VM-Consolidation"
        / "data"
        / "bitbrains_cloudsim_predictions.csv"
    )

    # Dashboard storage
    dashboard_runs_dir: Path = Path(__file__).parent.parent / "runs"
    dashboard_uploads_dir: Path = Path(__file__).parent.parent / "uploads"

    # Server
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000

    # Simulation
    sim_execution_enabled: bool = True
    sim_timeout_seconds: int = 3600

    # Optional large raw workload input processed directly by the backend.
    raw_bitbrains_csv: Path = (
        PROJECT_ROOT / "patch_models" / "bitbrains_raw_combined.csv"
    )

    # Model artifacts
    patchtst_checkpoint: Path = PROJECT_ROOT / "patch_models" / "PatchTST_best.pt"
    scaler_pkl: Path = PROJECT_ROOT / "patch_models" / "train_only_scaler.pkl"
    lr_classifier_pkl: Path = (
        PROJECT_ROOT / "patch_models" / "adaptive_logistic_regression.joblib"
    )
    lr_risk_threshold: float = 0.66
    classifier_sklearn_version: str = "1.6.1"

    def ensure_dirs(self) -> None:
        self.dashboard_runs_dir.mkdir(parents=True, exist_ok=True)
        self.dashboard_uploads_dir.mkdir(parents=True, exist_ok=True)

    def policy_dir(self, policy: str) -> Path:
        """Return the saved-results directory for a given policy (v10)."""
        p = policy.upper()
        if p == "ML_PROACTIVE":
            return self.sim_saved_results_dir / "ml_proactive"
        return self.sim_saved_results_dir / "comparison" / p.lower()

    def run_output_dir(self, run_id: str) -> Path:
        return self.dashboard_runs_dir / run_id / "output"

    def run_policy_dir(self, run_id: str, policy: str) -> Path:
        """Return a policy's output directory for a dashboard run."""
        selected = policy.upper()
        output_root = self.run_output_dir(run_id)
        if selected == "ML_PROACTIVE":
            return output_root
        return output_root / "comparison" / selected.lower()

    def resolve_project_path(self, path: Path) -> Path:
        """Resolve artifact paths relative to the project root unless absolute."""
        return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()

    @property
    def patchtst_checkpoint_path(self) -> Path:
        return self.resolve_project_path(self.patchtst_checkpoint)

    @property
    def scaler_path(self) -> Path:
        return self.resolve_project_path(self.scaler_pkl)

    @property
    def lr_classifier_path(self) -> Path:
        return self.resolve_project_path(self.lr_classifier_pkl)

    @property
    def raw_bitbrains_csv_path(self) -> Path:
        return self.resolve_project_path(self.raw_bitbrains_csv)


settings = Settings()
settings.ensure_dirs()

SUPPORTED_POLICIES = ["ML_PROACTIVE", "REACTIVE", "NO_MIGRATION", "MMT", "PABFD"]
