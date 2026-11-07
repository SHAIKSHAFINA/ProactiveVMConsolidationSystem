# PatchXProCon

PatchXProCon is a research application for VM workload forecasting and simulated consolidation. It combines a FastAPI backend, React/Vite dashboard, trained PatchTST inference, risk classification, and an isolated Java/Maven CloudSim Plus simulator.

## Repository layout

- `PatchXProCon-Dashboard/` — FastAPI backend and React dashboard.
- `PatchTST-VM-Consolidation/` — CloudSim Plus simulator source.
- `model_patch.ipynb` — training experiments and saved evaluation outputs.

## Data, models, and generated results

This repository intentionally excludes Bitbrains workload/prediction CSVs, model checkpoints/scalers/classifier binaries, uploads, and run outputs. These files may be large or subject to dataset/model distribution restrictions.

To enable raw-workload inference, place the authorized artifacts under `patch_models/` at the workspace root:

- `PatchTST_best.pt`
- `train_only_scaler.pkl`
- `adaptive_logistic_regression.joblib`

The backend expects the classifier artifact to be compatible with its configured 48-feature input. The notebook's saved classifier evaluation uses 57 features, so those classification metrics should not be attributed to the deployed artifact without reconciling its provenance. PatchTST notebook metrics are recorded experiment results, not a fresh evaluation of an artifact downloaded by a new user.

For simulator runs, provide compatible prediction input at the configured path or use the dashboard's upload flow. Prepared predictions are used as recorded; they are not fresh model inference.

## Local setup

See [the dashboard README](PatchXProCon-Dashboard/README.md) for backend/frontend setup, API details, and workflow instructions. The default configuration expects the simulator source under `PatchTST-VM-Consolidation/` at the workspace root. Copy `PatchXProCon-Dashboard/.env.example` to `PatchXProCon-Dashboard/backend/.env` and adjust local paths as needed. Do not commit `.env` or private datasets/model artifacts.

## Scientific limitations

- The application evaluates policies in simulation; it does not operate real cloud infrastructure.
- Host capacities are assumptions; disk I/O is not a placement constraint.
- Overload intervals are simulation proxies, not measured SLA violations.
- Energy is an approximate estimate, not metered facility energy or evidence of savings.
- See the [technical deep dive](PatchXProCon-Dashboard/PatchXProCon_Technical_Deep_Dive.md) for implementation details, evaluation numbers, and provenance caveats.
