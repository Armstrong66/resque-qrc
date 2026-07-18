# ResQue QRC Skill
## GIC 2026 | Dynamic Systems Forecasting | Track B: Weather Forecasting

> **This is a qBraid agent-executable skill package.**
> An AI coding agent can use this file to navigate the codebase, configure
> the quantum reservoir, run training, perform sweeps, and reproduce all
> results end-to-end without human intervention.

---

## SKILL OVERVIEW

**What this skill does:**
Runs a complete Quantum Reservoir Computing (QRC) pipeline for multi-output
atmospheric forecasting using NOAA ISD weather station data. The pipeline
implements a novel warm-start QRC architecture (ESN readout weight transfer)
combined with noise-assisted generalization, benchmarked against classical
baselines across qubit counts from 5 to 20.

**Repository root:** `resque_qrc/`
**Entry point:** `python main.py`
**Config file:** `config.py` (all hyperparameters — edit here only)
**Primary outputs:** `outputs/results/`

---

## ENVIRONMENT SETUP

### On qBraid Lab (use pre-installed quantum Python environment):
```bash
pip install -r requirements.txt
```

### Verify:
```bash
python -c "import pennylane; print('PennyLane OK:', pennylane.__version__)"
python -c "import torch; print('PyTorch OK, CUDA:', torch.cuda.is_available())"
python -c "from data.downloader import download_all; print('Pipeline imports OK')"
```

---

## STEP-BY-STEP REPRODUCTION

### Step 1 — Download data
```bash
python -c "
from data.downloader import download_all
paths = download_all()
print(f'Downloaded {len(paths)} files')
"
```
**What it does:** Downloads NOAA ISD global-hourly CSV files for station
`config.STATION_ID` (default `63450099999`, Addis Ababa Bole, Ethiopia), years 2018–2024, to `outputs/raw/AddisAbaba_Bole/`.
No API key required. Skips existing files automatically.

---

### Step 2 — Parse and preprocess
```bash
python -c "
from data.downloader import download_all
from data.parser import load_and_merge
from preprocessing.pipeline import WeatherPreprocessor

paths = download_all()
df = load_and_merge(paths)
prep = WeatherPreprocessor(df)
datasets = prep.build_all()
prep.save(datasets)
print('Datasets built:', {h: ds.summary() for h, ds in datasets.items()})
"
```
**What it does:** Parses ISD CSV fields (TMP, DEW, SLP, WND), derives relative
humidity via Magnus formula, resamples to 6-hourly, normalises (train-only
z-score), builds sliding windows (L=20), splits 70/15/15.

---

### Step 3 — Run Hamiltonian sweep (select J*, h*)
```bash
python -c "
from preprocessing.pipeline import WeatherPreprocessor
import pickle

ds = WeatherPreprocessor.load(6)
from experiments.sweeps import hamiltonian_sweep
J, h, df = hamiltonian_sweep(ds.X_train, ds.y_train, ds.X_val, ds.y_val)
print(f'Best: J*={J}, h*={h}')
"
```
**Output:** `outputs/results/hamiltonian_sweep.csv`, `best_hamiltonian.json`
**Expected:** J* ∈ [0.5, 1.5], h* ∈ [0.3, 0.8] for 6h horizon

---

### Step 4 — Run noise sweep (select p*)
```bash
python -c "
import json
from preprocessing.pipeline import WeatherPreprocessor
from experiments.sweeps import noise_sweep

ds = WeatherPreprocessor.load(6)
cfg = json.load(open('outputs/results/best_hamiltonian.json'))
p, df = noise_sweep(ds.X_train, ds.y_train, ds.X_val, ds.y_val,
                    J=cfg['J_star'], h=cfg['h_star'])
print(f'Optimal noise rate p*={p}')
"
```
**Output:** `outputs/results/noise_sweep.csv`, `best_noise.json`
**Interpretation:** If p* > 0, noise-assisted generalization is confirmed.

---

### Step 5 — Qubit scaling study
```bash
python -c "
import json
from preprocessing.pipeline import WeatherPreprocessor
from experiments.sweeps import qubit_scaling_study

ds = WeatherPreprocessor.load(6)
jh = json.load(open('outputs/results/best_hamiltonian.json'))
jn = json.load(open('outputs/results/best_noise.json'))
df = qubit_scaling_study(ds.X_train, ds.y_train, ds.X_val, ds.y_val,
                          J=jh['J_star'], h=jh['h_star'], p=jn['p_star'])
print(df)
"
```
**Output:** `outputs/results/qubit_scaling.csv`
**Challenge requirement:** Performance must be characterised across n = 5–20 qubits.

---

### Step 6 — Train classical baselines
```bash
python -c "
from preprocessing.pipeline import WeatherPreprocessor
from baselines.classical import run_persistence, run_arima, run_esn, run_rnn, ARIMA_AVAILABLE
from config import WINDOW_SIZE, TARGETS

ds = WeatherPreprocessor.load(6)

# Persistence
r = run_persistence(ds.y_val, ds.y_test, ds.X_val, ds.X_test, WINDOW_SIZE)
print('Persistence test RMSE:', r.test_rmse)

# ARIMA — prefers pmdarima, falls back to a statsmodels grid search if
# pmdarima is not installed (ARIMA_AVAILABLE=False only if NEITHER is present)
r = run_arima(ds.y_train, ds.y_val, ds.y_test, TARGETS)
print('ARIMA test RMSE:', r.test_rmse if r else f'unavailable (ARIMA_AVAILABLE={ARIMA_AVAILABLE})')

# ESN (also the default warm-start source for the QRC readout)
r_esn, fitted_esn = run_esn(ds.X_train, ds.y_train,
                              ds.X_val, ds.y_val,
                              ds.X_test, ds.y_test)
print('ESN test RMSE:', r_esn.test_rmse)

# LSTM (change model_type='gru' to run GRU) — returns (result, warm_start_extractor)
r_lstm, lstm_extractor = run_rnn(ds.X_train, ds.y_train, ds.X_val, ds.y_val,
                                  ds.X_test, ds.y_test, window=WINDOW_SIZE, model_type='lstm')
print('LSTM test RMSE:', r_lstm.test_rmse if r_lstm else 'unavailable')
"
```

---

### Step 7 — Train warm-start QRC
```bash
python -c "
import json
from preprocessing.pipeline import WeatherPreprocessor
from reservoir.quantum_reservoir import IsingQRC
from readout.ridge_readout import RidgeReadout
from baselines.classical import run_esn, run_rnn
from config import TARGETS, WARM_START_SOURCE, USE_DATA_REUPLOADING

ds = WeatherPreprocessor.load(6)
jh = json.load(open('outputs/results/best_hamiltonian.json'))
jn = json.load(open('outputs/results/best_noise.json'))

# Warm-start source is a ONE-LINE config switch — config.WARM_START_SOURCE
# is 'esn' | 'lstm' | 'gru' (NOT 'arima': ARIMA has no reservoir-like hidden
# state, so it structurally cannot warm-start a ridge readout).
if WARM_START_SOURCE == 'esn':
    _, fitted = run_esn(ds.X_train, ds.y_train, ds.X_val, ds.y_val, ds.X_test, ds.y_test)
    X_train_warm = fitted.get_reservoir_states(ds.X_train)
else:
    _, fitted = run_rnn(ds.X_train, ds.y_train, ds.X_val, ds.y_val, ds.X_test, ds.y_test,
                        model_type=WARM_START_SOURCE)
    X_train_warm = fitted.get_hidden_states(ds.X_train)

# Run quantum reservoir (best_hamiltonian.json records which encoding it was
# swept under — re-run Step 3 if USE_DATA_REUPLOADING has changed since)
qrc = IsingQRC(n_qubits=9, J=jh['J_star'], h=jh['h_star'],
               noise_rate=jn['p_star'], use_data_reuploading=USE_DATA_REUPLOADING)
H_train = qrc.run_sequence(ds.X_train)
H_val   = qrc.run_sequence(ds.X_val)

# Fit readout (auto-selects best strategy)
readout = RidgeReadout(target_names=TARGETS, warm_start=True)
best = readout.fit(H_train, ds.y_train[:len(H_train)],
                   H_val,   ds.y_val[:len(H_val)],
                   X_train_warm_start=X_train_warm[:len(H_train)])
readout.save_selection_log()
print('Selected strategy:', best.strategy, 'val_rmse:', best.val_rmse_mean)
"
```

---

### Step 8 — Full pipeline (all steps, single command)
```bash
python main.py
```

### Step 9 — Full pipeline, smoke test (fast, ~5 min)
```bash
python main.py --smoke_test
```

---

## AGENT-DRIVEN HYPERPARAMETER SWEEPS

### Custom qubit count sweep
```python
# Modify config.py QUBIT_COUNTS, then:
from experiments.sweeps import qubit_scaling_study
# ... load ds, then:
df = qubit_scaling_study(ds.X_train, ds.y_train, ds.X_val, ds.y_val,
                          J=J_star, h=h_star, p=p_star,
                          qubit_counts=[5, 7, 9, 12, 16, 20])
```

### Custom noise sweep
```python
from experiments.sweeps import noise_sweep
p_star, df = noise_sweep(..., noise_rates=[0, 0.001, 0.005, 0.01, 0.02, 0.05])
```

### Switch LSTM → GRU
```python
r, extractor = run_rnn(..., model_type='gru')   # One-line change
```

### Switch topology
```python
qrc = IsingQRC(n_qubits=9, topology='all_to_all')  # vs 'chain'
```

### Switch the QRC warm-start source (ESN / LSTM / GRU)
```python
# config.py: WARM_START_SOURCE = "esn"   # or "lstm" / "gru" — one-line change.
# main.py's _resolve_warm_start_states() and the notebook's Step 6/7 cells
# read this value directly; no other code needs to change.
# NOT valid: "arima" — ARIMA has no reservoir-like hidden state to transfer,
# and raises a clear ValueError if selected.
```

### Toggle data-reuploading encoding
```python
# config.py: USE_DATA_REUPLOADING = True   # or False for standard single-injection
# Re-run Step 3 (hamiltonian_sweep) afterwards — J*/h* are encoding-dependent
# and best_hamiltonian.json records which encoding produced them; main.py
# warns loudly if a cached sweep result predates the current setting.
qrc = IsingQRC(n_qubits=9, use_data_reuploading=True)   # or pass explicitly
```

### Change forecast horizon
```python
ds = WeatherPreprocessor.load(24)   # Load 24h horizon dataset
```

---

## OUTPUT FILES REFERENCE

| File | Description | Format |
|---|---|---|
| `outputs/raw/AddisAbaba_Bole/*.csv` | Raw NOAA ISD downloads | CSV |
| `outputs/processed/AddisAbaba_Bole/weather_6h.parquet` | Cleaned, resampled data | Parquet |
| `outputs/processed/AddisAbaba_Bole/dataset_h6.pkl` | Windowed dataset (6h) | Pickle |
| `outputs/processed/AddisAbaba_Bole/dataset_h24.pkl` | Windowed dataset (24h) | Pickle |
| `outputs/results/hamiltonian_sweep.csv` | (J, h) sweep RMSE grid | CSV |
| `outputs/results/best_hamiltonian.json` | Selected J*, h* (+ which encoding produced them) | JSON |
| `outputs/results/h{N}/baseline_status.json` | ok/skipped/failed + reason per baseline | JSON |
| `outputs/results/noise_sweep.csv` | Noise rate vs RMSE | CSV |
| `outputs/results/best_noise.json` | Selected p* | JSON |
| `outputs/results/qubit_scaling.csv` | RMSE vs qubit count | CSV |
| `outputs/results/shot_ablation.csv` | RMSE vs shot budget | CSV |
| `outputs/results/topology_comparison.csv` | Chain vs all-to-all | CSV |
| `outputs/results/readout_selection.json` | Auto-selected strategy | JSON |
| `outputs/results/results_h6.csv` | Full benchmark table (6h) | CSV |
| `outputs/results/results_h24.csv` | Full benchmark table (24h) | CSV |
| `outputs/results/warm_start_qrc_readout.pkl` | Fitted readout weights | Pickle |
| `outputs/results/warm_start_qrc_config.json` | Full reproducibility config | JSON |
| `outputs/logs/run_*.log` | Timestamped run logs | Text |

---

## CODEBASE NAVIGATION MAP

| Task | Module | Key function/class |
|---|---|---|
| Change any hyperparameter | `config.py` | Edit directly |
| Download data | `data/downloader.py` | `download_all()` |
| Parse ISD CSV | `data/parser.py` | `load_and_merge()` |
| Preprocess | `preprocessing/pipeline.py` | `WeatherPreprocessor` |
| Run reservoir | `reservoir/quantum_reservoir.py` | `IsingQRC.run_sequence()` |
| Qubit sweep | `reservoir/quantum_reservoir.py` | `run_qubit_sweep()` |
| Encoding ablation | `reservoir/quantum_reservoir.py` | `encoding_ablation()` |
| Fit readout | `readout/ridge_readout.py` | `RidgeReadout.fit()` |
| Warm-start init (any source) | `readout/ridge_readout.py` | `classical_warm_start_weights()` |
| Switch warm-start source | `config.py` | `WARM_START_SOURCE = "esn"\|"lstm"\|"gru"` |
| ESN baseline | `baselines/classical.py` | `run_esn()` |
| LSTM/GRU baseline | `baselines/classical.py` | `run_rnn(model_type=...)` |
| ARIMA baseline | `baselines/classical.py` | `run_arima()` (pmdarima, else statsmodels fallback) |
| Hamiltonian sweep | `experiments/sweeps.py` | `hamiltonian_sweep()` |
| Noise sweep | `experiments/sweeps.py` | `noise_sweep()` |
| Qubit scaling | `experiments/sweeps.py` | `qubit_scaling_study()` |
| Shot ablation | `experiments/sweeps.py` | `shot_ablation()` |
| Compute metrics | `evaluation/metrics.py` | `build_results_table()` |
| Full pipeline | `main.py` | `run_pipeline(args)` |
| Agentic task interface | `agent_runner.py` | `run_task(task, cfg)` |
| Reproducibility check | `verify_results.py` | `verify_all()` |
| Real-hardware validation | `scripts/hardware_validation.py` | `run_hardware_validation()` |
| Switch hardware backend | `reservoir/quantum_reservoir.py` | `IsingQRC(hardware_backend=...)` or `$QRC_BACKEND` |
| Aquila physical mapping (PRIMARY) | `reservoir/aquila_backend.py` | `AquilaBackend`, `_compute_geometry()` |
| Switch Aquila real hw vs. free emulator | `config.py` | `AQUILA_SUBMIT_TARGET` or `$AQUILA_SUBMIT_TARGET` |

---

## DEBUGGING

### Common issues and fixes

**PennyLane not found:**
```bash
pip install pennylane pennylane-lightning
```

**ARIMA missing from results_h*.csv:**
Check `outputs/results/h{N}/baseline_status.json` — it records exactly why
each baseline was skipped. ARIMA has two independent backends
(`baselines/classical.py`): pmdarima (preferred) and a statsmodels grid-search
fallback. It only truly disappears if NEITHER is importable:
```bash
pip install statsmodels pmdarima
```
The pipeline logs an ERROR (not a buried warning) at startup and per-horizon
whenever ARIMA is unavailable, specifically so this cannot go unnoticed.

**PyTorch CUDA not available (RTX workstation):**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

**Reservoir runs very slowly:**
- Reduce `QUBIT_COUNTS` in `config.py` for quick tests
- Use `--smoke_test` flag for fast iteration
- On qBraid Lab: ensure GPU-enabled kernel is selected

**Memory error on large sweep:**
- Reduce `J_SWEEP` and `H_SWEEP` in `config.py` to fewer values
- Run sweeps one at a time with `--skip_sweeps` and manual calls

**Download fails (NOAA server):**
- Check network; NOAA ISD endpoint is free but occasionally rate-limits
- Retries are automatic (3 attempts with backoff)
- Cached files are never re-downloaded

---

## REPRODUCING HEADLINE RESULTS

To reproduce the primary benchmark table used in the Phase 2 submission:

```bash
# 1. Full pipeline (first time, downloads data)
python main.py

# 2. Re-run with cached data (reproducibility check)
python main.py --skip_download

# 3. Check outputs
cat outputs/results/results_h6.csv
cat outputs/results/readout_selection.json
```

Expected output file: `outputs/results/results_h6.csv`
Contains: RMSE and MAE per target variable, VPT (Lyapunov-normalised),
for all models: Persistence, ARIMA, ESN, LSTM, GRU,
Cold-Start QRC, Warm-Start QRC.

---

## HARDWARE NOTES (Phase 3)

All sweeps (`hamiltonian_sweep`, `noise_sweep`, `qubit_scaling_study`,
`shot_ablation`, `topology_comparison`) run on simulator ONLY, by design —
each sweeps dozens of configs over hundreds of timesteps, and every timestep
is one circuit execution; pointed at real hardware that is thousands of
individually-queued jobs. Real hardware is for validating the FINAL,
already-selected config over a small subsampled window — see
`scripts/hardware_validation.py` and `docs/PROJECT_CRITIQUE.md` §3.2.

**Backend selection:** `IsingQRC(hardware_backend=...)`, or set `$QRC_BACKEND`
(`simulation` | `aquila` | `ibm`, default `simulation`).

**QuEra Aquila — PRIMARY, implemented:**
```bash
pip install bloqade-analog   # pulls in amazon-braket-sdk (Aquila is accessed via AWS Braket)
python scripts/hardware_validation.py --horizon 6 --n_steps 10 --backend aquila
```
Aquila is an *analog* Rydberg device with no gate set, so this is **NOT** a
device swap the way IBM is — `reservoir/aquila_backend.py::AquilaBackend`
re-expresses J/h as a real physical program: atom spacing sets the
interaction strength (Rydberg blockade), a global Rabi drive sets the
transverse field, and per-atom local detuning encodes the classical input
(the ONLY per-atom-addressable channel real hardware exposes — and it's
constrained non-negative, so encoding angles are affine-remapped from
`[-π,π]` into `[0,1]` coefficients). Measurement is Z-basis only; the
X-feature half comes from a basis-rotation pulse appended before
measurement, which is why this backend submits TWO programs per timestep.
**Read `reservoir/aquila_backend.py`'s module docstring in full before
trusting numbers from this backend** — it states every design choice and
its physical justification explicitly.

Validated end-to-end against TWO free local emulators (no credentials
needed) — `config.AQUILA_SUBMIT_TARGET` / `$AQUILA_SUBMIT_TARGET` controls
which:
- `"local_emulator"` (default) — Bloqade's own Python emulator.
- `"braket_local_emulator"` — AWS Braket's stricter local AHS simulator,
  closer to what real hardware submission validates against; this is what
  caught a real waveform-duration bug during development (see
  `docs/PROJECT_CRITIQUE.md`).
- `"aquila"` — **real hardware.** Submits actual circuits and consumes real
  qBraid/QuEra credits. **Not yet run against live Aquila hardware** in
  development. Start with `--n_steps 5-10`, not 50, the first time.

Geometry has hard limits from Aquila's published capabilities (256 atoms,
75×76 μm lattice, 4 μm minimum spacing) — `AquilaBackend` clamps atom
spacing to fit and raises a clear `ValueError` (not a silently-wrong
program) if a requested `n_qubits` doesn't fit even at minimum spacing.
Note: at this project's default calibration constants, `n_qubits=20` only
fits as `topology="all_to_all"` (a compact-ring approximation — true
uniform all-to-all coupling isn't physically realizable with 1/r^6
interactions), not `"chain"`.

**IBM Eagle/Heron — FALLBACK, implemented:**
```bash
pip install qiskit-ibm-runtime pennylane-qiskit
# Save an IBM Quantum account token once:
python -c "from qiskit_ibm_runtime import QiskitRuntimeService; \
           QiskitRuntimeService.save_account(channel='ibm_quantum', token='YOUR_TOKEN')"
python scripts/hardware_validation.py --horizon 6 --n_steps 30 --backend ibm
```
This IS a genuine drop-in device swap — the existing RY/IsingZZ/RX gate
circuit runs unchanged (see `reservoir/quantum_reservoir.py::_get_ibm_device`).
**Not yet executed against live IBM hardware** in development — confirm
end-to-end (or at minimum against a Qiskit simulator backend) before
treating its output as trustworthy for a submission.

---

## AGENTIC TASK INTERFACE

`agent_runner.py` is the machine-parseable entry point described in
`docs/AGENTIC_DESIGN_GUIDE.md`: one task per pipeline step, JSON config
overrides, `AGENT_RESULT: {...}` on the last line of stdout, and a mirrored
copy written to `outputs/results/agent_tasks/{task}.json`.

```bash
python agent_runner.py --task setup
python agent_runner.py --task hamiltonian_sweep --config '{"n_qubits": 9}'
python agent_runner.py --task train_baselines
python agent_runner.py --task train_qrc
python agent_runner.py --task benchmark_all
python agent_runner.py --task hardware_validation --config '{"horizon": 6, "n_steps": 50}'
python agent_runner.py --task full_run          # all of the above, in order
```

`verify_results.py` is the reproducibility check judges run after
re-executing the pipeline: confirms expected files exist, results tables
contain no NaN, and (once `--save_reference` has been run once after a
trusted full run) that key metrics match a stored reference within
tolerance.
```bash
python verify_results.py --save_reference   # do this ONCE after a trusted run
python verify_results.py                    # do this on every subsequent run
```

---

*ResQue | GIC 2026 | AI assistance disclosed per challenge rules.*
