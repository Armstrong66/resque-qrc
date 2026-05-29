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
63401099999 (Addis Ababa Bole, Ethiopia), years 2018–2024, to `outputs/raw/AddisAbaba_Bole/`.
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
from baselines.classical import run_persistence, run_esn, run_rnn
from config import WINDOW_SIZE, TARGETS

ds = WeatherPreprocessor.load(6)

# Persistence
r = run_persistence(ds.y_val, ds.y_test, ds.X_val, ds.X_test, WINDOW_SIZE)
print('Persistence test RMSE:', r.test_rmse)

# ESN (also generates warm-start weights)
r_esn, fitted_esn = run_esn(ds.X_train, ds.y_train,
                              ds.X_val, ds.y_val,
                              ds.X_test, ds.y_test)
print('ESN test RMSE:', r_esn.test_rmse)

# LSTM (change model_type='gru' to run GRU)
r_lstm = run_rnn(ds.X_train, ds.y_train, ds.X_val, ds.y_val,
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
from baselines.classical import run_esn
from config import TARGETS

ds = WeatherPreprocessor.load(6)
jh = json.load(open('outputs/results/best_hamiltonian.json'))
jn = json.load(open('outputs/results/best_noise.json'))

# Get ESN warm-start states
_, fitted_esn = run_esn(ds.X_train, ds.y_train,
                         ds.X_val, ds.y_val,
                         ds.X_test, ds.y_test)
X_train_esn = fitted_esn.get_reservoir_states(ds.X_train)

# Run quantum reservoir
qrc = IsingQRC(n_qubits=9, J=jh['J_star'], h=jh['h_star'],
               noise_rate=jn['p_star'])
H_train = qrc.run_sequence(ds.X_train)
H_val   = qrc.run_sequence(ds.X_val)

# Fit readout (auto-selects best strategy)
readout = RidgeReadout(target_names=TARGETS, warm_start=True)
best = readout.fit(H_train, ds.y_train[:len(H_train)],
                   H_val,   ds.y_val[:len(H_val)],
                   X_train_esn=X_train_esn[:len(H_train)])
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
r = run_rnn(..., model_type='gru')   # One-line change
```

### Switch topology
```python
qrc = IsingQRC(n_qubits=9, topology='all_to_all')  # vs 'chain'
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
| `outputs/results/best_hamiltonian.json` | Selected J*, h* | JSON |
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
| Fit readout | `readout/ridge_readout.py` | `RidgeReadout.fit()` |
| Warm-start init | `readout/ridge_readout.py` | `esn_warm_start_weights()` |
| ESN baseline | `baselines/classical.py` | `run_esn()` |
| LSTM/GRU baseline | `baselines/classical.py` | `run_rnn(model_type=...)` |
| ARIMA baseline | `baselines/classical.py` | `run_arima()` |
| Hamiltonian sweep | `experiments/sweeps.py` | `hamiltonian_sweep()` |
| Noise sweep | `experiments/sweeps.py` | `noise_sweep()` |
| Qubit scaling | `experiments/sweeps.py` | `qubit_scaling_study()` |
| Shot ablation | `experiments/sweeps.py` | `shot_ablation()` |
| Compute metrics | `evaluation/metrics.py` | `build_results_table()` |
| Full pipeline | `main.py` | `run_pipeline(args)` |

---

## DEBUGGING

### Common issues and fixes

**PennyLane not found:**
```bash
pip install pennylane pennylane-lightning
```

**ARIMA import error:**
```bash
pip install pmdarima statsmodels
```

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

**QuEra Aquila (Rydberg analog):**
The IsingQRC reservoir maps directly to Aquila's Rydberg blockade Hamiltonian.
Install Bloqade on qBraid Lab (pre-installed in Bloqade environment):
```bash
# On qBraid Lab — switch kernel to "Bloqade" environment
# Then follow: github.com/QuEraComputing/QRC-tutorials
```
The `IsingQRC` class is designed for easy extension to Bloqade submission.

**IBM Eagle/Heron (fallback):**
```bash
pip install qiskit qiskit-ibm-runtime
# Use qBraid SDK for provider-agnostic submission
```

---

*ResQue | GIC 2026 | AI assistance disclosed per challenge rules.*
