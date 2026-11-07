# PatchXProCon Dashboard

PatchXProCon is a local, dataset-driven research application for validating prediction CSVs and running the frozen CloudSim Plus simulator in isolated copies. On startup it restores the last selected dataset and its latest completed run when available; otherwise it opens dataset upload.

## Features

- Two main destinations: **Run Pipeline** and **Results**. Raw Bitbrains traces use the configured PatchTST and Logistic Regression artifacts before CloudSim; prepared prediction CSVs use their recorded forecast/risk columns without fresh inference.
- FastAPI endpoints for policy summaries, comparisons, host snapshots, VM placement and details, migrations, decisions, dataset catalog/upload, and asynchronous runs.
- A separate `POST /api/dataset/infer` operation generates fresh forecasts for a compatible raw dataset without starting CloudSim.
- Fresh inference accepts batches of up to 50 selected VMs and checkpoints completed VMs in a per-dataset SQLite store. Later batches and CloudSim runs resume from those per-VM checkpoints, then merge predictions by `(vm_id, decision_time)` before simulation.
- Interactive VM and local-time-window forecast exploration, plus configurable simulation VM/time subsets and host assumptions. Every new simulation evaluates all five policies against the same selected input.
- Interactive CPU/memory and disk forecast charts with current values, risk probability, threshold, and alert outcome. Raw and prepared prediction modes are identified separately.
- Searchable policy/event/decision views with pagination and CSV exports.
- Initial drag-and-drop CSV upload with schema, timestamp-order, missing-value, finite-number, numeric-range validation, and preview.
- Asynchronous simulation runs that use a unique, private copy of the simulator project and selected CSV for each run.

## Data locations and safety

The backend reads the existing simulator and v10 outputs using the configured paths in `backend/.env`:

- `SIM_PROJECT_DIR=D:/PatchTST-VM-Consolidation/PatchTST-VM-Consolidation`
- `SIM_SAVED_RESULTS_DIR=D:/PatchTST-VM-Consolidation/PatchTST-VM-Consolidation/results_v10`
- `SIM_INPUT_CSV=D:/PatchTST-VM-Consolidation/PatchTST-VM-Consolidation/data/bitbrains_cloudsim_predictions.csv`

For every simulation, the backend copies `pom.xml`, `src/`, optional `.mvn/`, and the selected input to `backend/runs/<run_id>/simulator/`. Raw workloads first resume or generate fresh predictions from per-VM inference checkpoints and select the complete requested VM/time window; prepared prediction CSVs are passed through as recorded inputs. The inference-only endpoint stores merged output separately under `backend/runs/inference/<dataset_id>/predictions.csv` and does not launch CloudSim. Maven compiles and executes only the isolated copy. Run outputs and logs are stored under `backend/runs/<run_id>/output/`; uploads and their metadata are stored under `backend/uploads/`. Existing simulator source, input CSV, and result folders are not changed.

The runner accepts only one active run per backend process. Use one Uvicorn worker. Each run is isolated from other runs and writes to a new directory. The Reset view control only clears the selected run view; it does not delete history or cancel a process. Completed run directories and the uploaded Bitbrains prepared-prediction baseline are retained.

## CSV upload contract

Required columns:

`vm_id`, `decision_time`, `current_cpu_pct`, `current_memory_kb`, `forecast_cpu_max`, `forecast_memory_max`, `risk_probability`, `risk_threshold`

Optional disk columns include `current_disk_read_kbps`, `current_disk_write_kbps`, `forecast_disk_read_max`, and `forecast_disk_write_max`.

Browser uploads remain limited to 200 MiB. Larger combined raw Bitbrains files can be processed in place by setting `RAW_BITBRAINS_CSV` to their path (default: `patch_models/bitbrains_raw_combined.csv`) and selecting **Use configured local CSV** on the Pipeline page. The backend validates the file in 100,000-row chunks and does not upload or copy the raw file. Its VM rows must be contiguous in the CSV to keep inference memory bounded. Fresh predictions are written to the isolated run directory.

Prepared prediction CSVs require `vm_id`, `decision_time`, current CPU/memory, forecast CPU/memory, and recorded risk probability/threshold. Raw Bitbrains CSVs require `Timestamp [ms]`, an identifier column (`VM_ID` or `vm_id`), and the six resource columns used by the training-time preprocessing; the timestamp values are interpreted as Unix seconds, as in the research notebook. CPU values above 100 are not rejected.

Raw workload inference uses a 96-step/four-feature input window, a 12-step/four-feature forecast, the exact training feature order, and a 48-feature classifier vector. Inference-only calls may select a VM batch and time window; successfully completed batch rows are merged and de-duplicated by VM and decision timestamp, allowing later batches to resume without rerunning completed VMs. The original model and scaler are never fitted or changed.

The Simulation Control Room can run one VM, a comma-separated selection of up to 100 VMs, or all VMs, and optionally restrict the inclusive prediction time window. Host count (10–500), host RAM (MiB), CPU cores/PEs, CPU/RAM overload limits, and simulation risk threshold are explicit per-run assumptions. Defaults are 200 hosts, 262,144 MB RAM, 32 PEs, 0.80 CPU/RAM limits, and a 0.66 risk threshold. Changing the simulation risk threshold updates only the threshold in that run's isolated simulator input; recorded probabilities and the source prediction CSV remain unchanged. A run creates a separate run ID and the frozen simulator produces results for all five policies. The selected "Primary results policy" controls which policy is foregrounded in the dashboard, not which comparison policies are executed.

Raw workload inference defaults to fresh model predictions; prepared prediction CSVs use their recorded forecasts and risk values without fresh inference. Simulation outputs (placement, migrations, estimated energy, and overload proxies) are separate from both prediction sources.

Selecting or uploading a dataset clears the prior result selection; Results show output only for a successful run associated with the active dataset ID and run ID. Earlier completed runs remain in run history and their isolated outputs remain available. The saved v10 baseline remains read-only and is not presented as a newly uploaded dataset's result.

### Local model artifacts

The default model paths are resolved from the workspace root and can be overridden with environment variables:

- `PATCHTST_CHECKPOINT=patch_models/PatchTST_best.pt`
- `LR_CLASSIFIER_PKL=patch_models/adaptive_logistic_regression.joblib`
- `SCALER_PKL=patch_models/train_only_scaler.pkl`
- `RAW_BITBRAINS_CSV=patch_models/bitbrains_raw_combined.csv`

The train-only scaler is mandatory: it is the `StandardScaler` fitted on 7,200,654 complete training rows in the research notebook. Do not fit a replacement scaler on an uploaded dataset. The classifier artifact was serialized with scikit-learn 1.6.1; the backend dependency is pinned to that version to avoid loading it under an incompatible runtime.

## Scientific limitations shown in the UI

- Energy is an approximate simulation estimate, not a measured data-center energy reading.
- Overload intervals are simulation proxies, not observed SLA violations.
- Host CPU and RAM capacities are simulation assumptions, not facts in the Bitbrains trace.
- Disk I/O constraints are `NOT_MODELED`.
- Saved v10 reports `powered_down_hosts = 0` for `ML_PROACTIVE`.
- Energy is an estimate and overload intervals are simulation proxies, not actual facility measurements or SLA violations.

## Start the backend on Windows

From the dashboard project:

```powershell
cd D:\PatchTST-VM-Consolidation\PatchXProCon-Dashboard\backend
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Use a Python version supported by the pinned dependencies. Java 17 and Maven must be installed and on `PATH` to run simulations. If Maven is unavailable, saved-result browsing still works and a submitted simulation reports an explicit error.

## Start the frontend

In a second terminal:

```powershell
cd D:\PatchTST-VM-Consolidation\PatchXProCon-Dashboard\frontend
npm install
npm run dev
```

Open http://localhost:5173/.

Backend API documentation is available at http://127.0.0.1:8000/api/docs.

## Validate the frontend

```powershell
cd D:\PatchTST-VM-Consolidation\PatchXProCon-Dashboard\frontend
npm run build
```

The frontend uses `VITE_API_BASE_URL` when configured and otherwise connects to `http://127.0.0.1:8000`.
