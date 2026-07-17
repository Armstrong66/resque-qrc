# ResQue — Agentic Pipeline & Deployment Design Guide
**ResQue | GIC 2026 Phase 3 | Internal Team Reference**

> This document standardises how the team builds, tests, and extends the agentic layer on top of the quantum forecasting pipeline. It covers three concerns: (1) the qBraid agentic skill layer judges will execute, (2) the Vercel live inference app, and (3) honest guidance on where simulator vs. real hardware fits post-competition. Read this before writing any new integration code.

---

## 1. What "agentic" means in this context

The challenge organisers want an AI coding agent to be able to:
- Read `skill/SKILL.md` and understand the codebase
- Configure the reservoir (change `config.py` parameters)
- Run training end-to-end without human intervention
- Reproduce specific results by re-running named experiment scripts
- Report metrics back in a structured format it can parse

This is not a general-purpose AI agent. It is a **structured, contract-driven interface** between the codebase and an automated executor. The contract is `SKILL.md`. Everything else (qBraid notebook, CLI, Vercel API) derives from the same underlying pipeline — the agentic layer is just a clean, machine-readable façade over it.

**Design principle: one pipeline, multiple interfaces.**

```
config.py  ←──── single source of truth for all parameters
    │
    ├── main.py              ← CLI (local / nohup / bash)
    ├── run_qbraid.ipynb     ← Notebook (judges / qBraid Lab)
    ├── agent_runner.py      ← Agentic interface (NEW — see §3)
    └── api/infer.py         ← Vercel serverless endpoint (see §5)
```

Nothing in the quantum pipeline should change depending on which interface called it. All four interfaces call the same preprocessing, reservoir, readout, and evaluation modules.

---

## 2. SKILL.md contract — what it must guarantee

The existing `skill/SKILL.md` is good structural documentation. For the agentic layer to work reliably, it must additionally satisfy these properties:

### 2.1 Every step must be independently re-runnable

Each step in `SKILL.md` must be a single Python command that can be run in isolation and produce a verifiable output file. If a step fails, the agent must be able to identify which step failed by checking which output file is missing.

**Pattern to follow in SKILL.md:**

```markdown
### Step N — [Name]
**Command:**
python -c "from module import fn; fn()"

**Produces:**
outputs/results/[filename].csv  OR  outputs/results/[filename].json

**Verify success:**
python -c "import json; d=json.load(open('outputs/results/[filename].json')); assert 'key' in d, 'Step N failed'"
```

Every step must have a `Produces` and `Verify success` line. An agent that cannot verify a step succeeded cannot safely proceed to the next one.

### 2.2 Agent entry points must accept structured JSON input

Add a thin wrapper `agent_runner.py` (see §3) that accepts a JSON config override so an agent can run specific experiments without editing any Python file:

```bash
python agent_runner.py --task hamiltonian_sweep --config '{"n_qubits": 9, "max_steps": 300}'
python agent_runner.py --task full_train --config '{"use_data_reuploading": true}'
python agent_runner.py --task encode_ablation
python agent_runner.py --task benchmark_all
```

### 2.3 All outputs must be machine-parseable

Every experiment must write a JSON summary alongside any CSV. Agents parse JSON, not table printouts. The JSON must contain at minimum:

```json
{
  "task": "hamiltonian_sweep",
  "status": "complete",
  "best_params": {"J": 0.3, "h": 0.8},
  "val_rmse_mean": 0.9214,
  "timestamp": "2026-07-15T14:32:01Z",
  "wall_clock_s": 312
}
```

### 2.4 The agent must be able to self-verify reproducibility

Add a `verify_results.py` script that reads all output JSONs and confirms:
- All expected files exist
- Key metrics match within tolerance of a stored reference run
- No NaN values in any result

This is what judges run after re-executing your code to confirm their results match yours.

---

## 3. `agent_runner.py` — the agentic interface

This file is the single point of entry for any automated executor. It reads a task name and optional JSON config, runs the appropriate pipeline module, and writes a structured result. The team's coder should build this.

**Interface contract:**

```python
# agent_runner.py
"""
Agentic interface for ResQue QRC pipeline.
Called by qBraid AI agent or any automated executor.

Usage:
    python agent_runner.py --task <task_name> [--config '<json_string>']

Tasks:
    setup              Install dependencies, verify environment
    download_data      Download NOAA ISD CSVs
    preprocess         Parse, clean, PCA, window, split
    eda                Run data quality audit
    hamiltonian_sweep  Sweep J x h, write best_hamiltonian.json
    noise_sweep        Sweep depolarizing rate, write best_noise.json
    encode_ablation    Standard vs. data reuploading comparison
    qubit_scaling      Scale n=5→20, write qubit_scaling.csv
    shot_ablation      Vary shot budget, write shot_ablation.csv
    train_baselines    Persistence + ARIMA + ESN + LSTM + GRU
    train_qrc          Cold-start and warm-start QRC
    benchmark_all      Full results tables both horizons
    verify_results     Check all outputs match reference run
    full_run           Runs all tasks in order (complete pipeline)
"""
```

**Key implementation notes for the coder:**

- Load config from `config.py` first, then apply JSON overrides on top — never edit `config.py` from within this script
- Wrap every task in try/except and write `{"status": "failed", "error": "..."}` to the output JSON on failure — agents must be able to read failure reasons
- Print `AGENT_RESULT: <json>` to stdout at the end of each task — this is how the calling agent reads the result without parsing log files
- Each task should be idempotent — running it twice should produce the same result, not crash on existing files

---

## 4. qBraid agentic execution — how it works in practice

qBraid's agentic coding environment will likely work as follows: it reads `SKILL.md`, identifies tasks, calls `agent_runner.py` for each, reads the JSON output, and decides what to run next. Your job is to make that sequence completely reliable.

### 4.1 Recommended task execution order in SKILL.md

```
setup → download_data → preprocess → eda
  → hamiltonian_sweep → noise_sweep → encode_ablation
  → train_baselines → qubit_scaling → shot_ablation
  → train_qrc → benchmark_all → verify_results
```

Each task writes its output before the next begins. If any task produces `"status": "failed"`, the agent stops and reports which task failed and why.

### 4.2 Hardware integration hook

The `train_qrc` task should check an environment variable to select backend:

```python
import os
BACKEND = os.environ.get('QRC_BACKEND', 'simulation')
# 'simulation'  → PennyLane default.qubit / lightning.qubit
# 'aquila'      → QuEra via Bloqade (requires qBraid Bloqade kernel)
# 'ibm'         → IBM Eagle via Qiskit Runtime (requires IBM token)
```

On qBraid, the organiser sets `QRC_BACKEND=aquila` or `QRC_BACKEND=ibm` before running — your code picks it up without any file edits. This is the clean hardware integration point.

### 4.3 Testing the agentic layer yourself

Before submission, the team must run this end-to-end test locally:

```bash
# Simulate what the qBraid agent will do
python agent_runner.py --task setup
python agent_runner.py --task download_data
python agent_runner.py --task preprocess
python agent_runner.py --task eda
python agent_runner.py --task hamiltonian_sweep --config '{"max_steps": 100}'
python agent_runner.py --task train_qrc --config '{"use_data_reuploading": true}'
python agent_runner.py --task benchmark_all
python agent_runner.py --task verify_results

# Each command should exit 0 and print AGENT_RESULT: {"status": "complete", ...}
```

If any command fails or produces unexpected output, fix it before submission. Judges running this sequence must get the same numbers as your write-up.

---

## 5. Vercel live inference app — design

This is forward-looking but feasible now. The Vercel app calls a serverless Python function that loads the trained QRC readout and runs inference on recent NOAA data. **Critically: the quantum reservoir is not run at inference time on Vercel.** The reservoir's expectation values from the most recent observation window are pre-computed (or the classical readout is called directly), because serverless functions have a 10-second timeout and PennyLane circuit evaluation takes much longer.

### 5.1 Architecture

```
Vercel Frontend (Next.js / React)
    │
    │  HTTP GET /api/forecast
    ▼
Vercel Serverless Function (api/infer.py or api/infer.js)
    │  Loads: trained W* matrix (ridge readout weights, ~100KB)
    │  Calls: fast classical feature computation OR cached reservoir states
    │  Returns: JSON forecast for next 6h and 24h
    ▼
Browser renders live chart of T, RH, P, wind over Addis Ababa
```

### 5.2 What runs where

| Component | Runs where | Reason |
|---|---|---|
| QRC reservoir (full) | qBraid / local GPU | Too slow for serverless; done offline |
| Ridge readout (W*) | Vercel serverless | Pure matrix multiply, sub-millisecond |
| NOAA data fetch | Vercel serverless | HTTP GET to NOAA NCEI endpoint |
| Feature preprocessing (PCA) | Vercel serverless | Fast numpy; PCA matrix loaded from file |
| Model weights (W*) | Vercel static file or KV store | ~100KB pickle/JSON, loaded on cold start |

### 5.3 Inference endpoint design

```python
# api/infer.py  (Vercel Python serverless function)
"""
GET /api/forecast?station=63450099999&horizon=6
Returns: {"timestamp": "...", "temperature": ..., "humidity": ...,
          "pressure": ..., "wind_speed": ..., "horizon_h": 6}
"""
import json
import numpy as np
import pickle
from pathlib import Path

# Loaded once at cold start (not per-request)
W_STAR    = pickle.loads(Path('model/W_star.pkl').read_bytes())
PCA_MEAN  = np.load('model/pca_mean.npy')
PCA_COMP  = np.load('model/pca_components.npy')
SCALER_M  = np.load('model/scaler_mean.npy')
SCALER_S  = np.load('model/scaler_std.npy')

def fetch_recent_observations(station_id: str, n_steps: int = 20) -> np.ndarray:
    """Fetch last n_steps * 6h of NOAA ISD data for the station."""
    # Call NOAA NCEI API or cache layer
    # Return normalised, PCA-reduced array of shape (n_steps, n_components)
    ...

def handler(request):
    station  = request.args.get('station', '63450099999')
    horizon  = int(request.args.get('horizon', 6))

    # 1. Get recent observations (last 5 days = 20 steps at 6h)
    X_window = fetch_recent_observations(station, n_steps=20)

    # 2. Apply shared PCA (same transform used in training)
    X_pca = (X_window - PCA_MEAN) @ PCA_COMP.T

    # 3. Matrix multiply with trained W* (the only "model" call)
    y_norm = X_pca[-1:] @ W_STAR     # shape (1, 4)

    # 4. Inverse-transform to physical units
    y_phys = y_norm * SCALER_S + SCALER_M

    targets = ['temperature', 'humidity', 'pressure', 'wind_speed']
    forecast = {t: round(float(y_phys[0, i]), 2) for i, t in enumerate(targets)}
    forecast.update({'horizon_h': horizon, 'station': station,
                     'backend': 'qrc_simulator'})
    return json.dumps(forecast)
```

### 5.4 Model export script

The team needs one script to export the trained model for Vercel:

```bash
python scripts/export_model.py
# Writes to model/
#   W_star.pkl           ← ridge readout weights
#   pca_mean.npy         ← PCA mean vector
#   pca_components.npy   ← PCA component matrix
#   scaler_mean.npy      ← z-score mean
#   scaler_std.npy       ← z-score std
#   model_config.json    ← metadata (n_qubits, J, h, horizon, etc.)
```

All files together should be under 1MB. The Vercel function loads them on cold start.

### 5.5 Real hardware post-competition

Honest assessment: running the QRC reservoir live on real hardware for every inference request is not feasible at any free or low-cost tier today. QuEra Aquila costs credits per shot; IBM Quantum has queue times of minutes to hours. For a live demo the options are:

| Option | Cost | Latency | Recommended for |
|---|---|---|---|
| Simulator backend (lightning.qubit) | Free | ~1–60s depending on n | Post-competition demo |
| Pre-computed reservoir states (current window cached, refreshed every 6h) | Free | Sub-millisecond | Live Vercel app |
| AWS Braket + QuEra (on-demand) | ~$0.10–1.00 per task | Minutes | Pitch demo only |
| IBM Quantum (free tier) | Free but queued | Hours | Not viable for live |

**Recommended approach**: the Vercel app uses the trained W* matrix for instant inference (the classical readout is the product), and describes itself as "powered by QRC trained on QuEra Aquila simulator" with the hardware validation result cited in the model card. Refresh the NOAA data fetch every 6 hours via a cron job. This is honest, fast, and deployable for free.

---

## 6. Team task assignments

| Role | Immediate Phase 3 task |
|---|---|
| **Coder** | Build `agent_runner.py` (§3); wire `QRC_BACKEND` env var into `quantum_reservoir.py`; build `scripts/export_model.py` |
| **Data/Technical Lead** | Run full pipeline end-to-end; fill benchmark tables; run encoding ablation; own `verify_results.py` |
| **Content/Knowledge Expert** | Write Phase 3 5-page paper; own the quantum advantage narrative and limitations section |
| **Business/Production** | Build Vercel serverless endpoint (`api/infer.py`); connect to frontend chart; coordinate QPU access request |
| **Project Manager** | End-to-end agentic test (§4.3); package `ResQue_Challenge_Phase3.zip`; README completeness check; deadline tracking |

---

## 7. Submission packaging checklist

```
ResQue_Challenge_Phase3.zip
├── GIC_2026_Cover_Page.docx          ← Official template, unmodified
├── ResQue_Phase3_Writeup.pdf   ← 5 pages, 11pt TNR, single spacing
├── README.md                        ← This is what judges read first
├── run_qbraid.ipynb                 ← Judges run this to reproduce results
├── skill/SKILL.md                   ← Agent-executable skill package
├── agent_runner.py                  ← Agentic interface (NEW)
├── verify_results.py                ← Reproducibility checker (NEW)
├── config.py
├── main.py
├── requirements.txt
├── data/
├── preprocessing/
├── reservoir/
├── readout/
├── baselines/
├── experiments/
├── evaluation/
├── model/                           ← Exported weights for Vercel (optional)
│   ├── W_star.pkl
│   ├── pca_components.npy
│   └── model_config.json
└── outputs/results/                 ← All CSV/JSON results from your run
    ├── results_h6.csv
    ├── results_h24.csv
    ├── encoding_ablation.json
    ├── qubit_scaling.csv
    ├── noise_sweep.csv
    ├── shot_ablation.csv
    └── warm_start_qrc_config.json
```

**Before zipping — run this:**

```bash
# 1. Clean agentic test (most important)
python agent_runner.py --task full_run
python agent_runner.py --task verify_results

# 2. Confirm notebook runs clean
jupyter nbconvert --to notebook --execute run_qbraid.ipynb --output run_qbraid_executed.ipynb

# 3. Confirm zip structure
zip -r ResQue_Challenge_Phase3.zip . \
    --exclude "*.pyc" --exclude "__pycache__/*" \
    --exclude "outputs/raw/*" --exclude ".git/*"
```

**Deadline: July 26, 2026 at 11:59 PM EST**

---

## 8. Principles to hold throughout

- **One pipeline, multiple interfaces.** Never duplicate logic between notebook, CLI, and agent runner. If you fix a bug in the pipeline, it's fixed everywhere.
- **Every claim needs a number.** The write-up must have qubit count, circuit depth, wall-clock time, and RMSE for every result claimed. Judges are explicitly told qualitative descriptions score zero.
- **Honest limitations outscore overstatement.** The rubric explicitly rewards teams who explain where QRC does and does not provide benefit. The ESN gap is real — explain it mechanistically and show the data reuploading result honestly, whether it closes the gap or not.
- **The README is the product.** A judge who can reproduce your results from the README in 15 minutes will score you higher than a team with impressive claims and a confusing README.
- **The Vercel app is a bonus, not a substitute.** A working reproducible qBraid notebook matters more for scoring. Build the app in parallel, not instead of.
