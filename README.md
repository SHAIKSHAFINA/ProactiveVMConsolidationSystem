# PatchXProCon — Predictive VM Consolidation Research System

<p align="center">
  <img src="https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React 18" />
  <img src="https://img.shields.io/badge/TypeScript-5.6-3178C6?style=for-the-badge&logo=typescript&logoColor=white" alt="TypeScript 5.6" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/PyTorch-PatchTST-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Java-CloudSim-orange?style=for-the-badge&logo=openjdk&logoColor=white" alt="Java and CloudSim" />
</p>

<img width="900" height="526" alt="image" src="https://github.com/user-attachments/assets/dc809c44-ab3d-4e4b-abe8-ff6ef2d9356f" />
<img width="900" height="543" alt="image" src="https://github.com/user-attachments/assets/07aa1694-ceb4-431a-8373-d82ff17d3034" />
<img width="900" height="536" alt="image" src="https://github.com/user-attachments/assets/48f81b8a-75b1-47b6-b4d9-d66c8623655c" />




<p align="center">
  <strong>A dataset-driven research dashboard for forecasting VM resource demand, estimating overload risk, and comparing consolidation policies in CloudSim Plus.</strong>
  <br />
  Combines a React/Vite interface, FastAPI inference and run APIs, trained PatchTST artifacts, and isolated Java/Maven simulation runs.
</p>

---

## Contents

- [Overview](#overview)
- [Workflow and architecture](#workflow-and-architecture)
- [Models and recorded results](#models-and-recorded-results)
- [Dataset and scale](#dataset-and-scale)
- [Simulator policies and outputs](#simulator-policies-and-outputs)
- [Technology](#technology)
- [Run locally](#run-locally)
- [Artifacts and data policy](#artifacts-and-data-policy)
- [Limitations and research caveats](#limitations-and-research-caveats)
- [Documentation](#documentation)

## Overview

PatchXProCon explores predictive virtual-machine consolidation using Bitbrains workload traces. For supported raw traces, the backend preprocesses per-VM resource histories, loads saved forecasting and classification artifacts, and supplies the resulting predictions to CloudSim Plus. The dashboard presents run-scoped forecasts, simulated placements, host utilization, policy summaries, and migration events.

The project is a **research and simulation tool**. It does not control or migrate VMs in a real cloud environment.

## Workflow and architecture

```text
Raw workload CSV
    → validation and per-VM chronological preprocessing
    → saved train-only scaler
    → PatchTST forecast (96 history steps → 12 forecast steps)
    → Logistic Regression-compatible risk probability
    → unique isolated CloudSim Plus run
    → run-scoped metrics, placements, decisions, and migration events
    → React dashboard and CSV exports
```

Prepared-prediction CSVs are a separate input mode: existing forecast and risk columns are passed through as recorded values; the app does not describe them as fresh inference. Each simulation gets its own run ID, input copy, output directory, and metadata so results are associated with the selected dataset and run.

## Models and recorded results

- **Forecasting model:** non-RevIN PatchTST, configured for 96 five-minute observations and a 12-step horizon across CPU, memory, disk read, and disk write.
- **Risk model:** Logistic Regression-compatible classifier artifact. The backend expects a 48-feature vector and applies its configured 0.66 risk cutoff.
- The notebook also evaluates DLinear and PatchTST+RevIN forecasts and compares several classifiers. Those experiments are not necessarily the artifacts loaded by the backend.

The notebook’s saved PatchTST results on its capped **100,000-window test sample**:

| Forecast target | MAE | RMSE | R² |
|---|---:|---:|---:|
| CPU usage | 1.7746 percentage points | 8.6970 percentage points | 0.7025 |
| Memory | 232,543 KB | 1,252,802 KB | 0.5641 |
| Disk read | 163.43 KB/s | 2,707.04 KB/s | 0.7188 |
| Disk write | 45.90 KB/s | 792.74 KB/s | 0.4455 |

These are **forecasting metrics**, not classification accuracy or recall. They are saved notebook experiment outputs, not a fresh evaluation of the artifact during this publish step.

> **Classifier provenance:** the notebook’s evaluated classifier builds 57 features, but the backend-loaded artifact expects 48. Until that discrepancy is reconciled, do not report the notebook’s classifier scores as verified performance for the deployed classifier.

## Dataset and scale

The documented local Bitbrains inputs include:

- Raw trace: **11,221,800 rows, 1,250 VM IDs**.
- Prepared prediction file: **456,799 rows, 1,217 VM IDs**.
- In the recorded raw inference run, **1,159 VMs** had a continuous segment long enough for inference; other VMs were not given fabricated predictions.

These files are not included in this repository. Use data only when its license and distribution terms permit it.

## Simulator policies and outputs

CloudSim Plus runs the selected input through five policies:

`ML_PROACTIVE` · `REACTIVE` · `NO_MIGRATION` · `MMT` · `PABFD`

Run outputs can include policy summaries, host utilization snapshots, VM placement, decision records, and migration lifecycle events. Simulation host capacities are explicit assumptions; placements represent simulator outcomes, not actual infrastructure changes.

## Technology

| Component | Technologies |
|---|---|
| Dashboard | React 18, TypeScript, Vite, Recharts |
| API and inference | FastAPI, Python, pandas, NumPy, PyTorch, scikit-learn/joblib |
| Simulation | Java, Maven, CloudSim Plus |
| Run/preprocessing state | CSV, JSON metadata, SQLite inference checkpoints |

## Run locally

1. Follow the [dashboard setup guide](PatchXProCon-Dashboard/README.md) for Python, Node.js, and Java/Maven prerequisites and start commands.
2. The default simulator source path is `PatchTST-VM-Consolidation/` beside this README.
3. Copy `PatchXProCon-Dashboard/.env.example` to `PatchXProCon-Dashboard/backend/.env` if you need local overrides.
4. To run raw inference, supply the authorized artifacts described below. Otherwise, use a compatible prepared-prediction CSV through the dashboard.

The backend’s configuration check requires Python packages from `PatchXProCon-Dashboard/backend/requirements.txt`; the frontend has a production build command, `npm run build`.




