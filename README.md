# ResQue — Warm-Start QRC for Multi-Output Atmospheric Forecasting
### GIC 2026 | Dynamic Systems Forecasting | Track B: Weather

[![Launch on qBraid](https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png)](https://account.qbraid.com?gitHubUrl=https://github.com/YOUR-ORG/resque-qrc)

---

## Project Structure

```
resque_qrc/
├── config.py                    ← Single source of truth (all hyperparameters)
├── main.py                      ← Full pipeline orchestrator
├── requirements.txt
├── README.md
│
├── utils/
│   └── __init__.py              ← Structured logger (used by every module)
│
├── data/
│   ├── downloader.py            ← NOAA ISD CSV download (no API key)
│   └── parser.py                ← ISD parser → clean DataFrame
│
├── preprocessing/
│   └── pipeline.py              ← Normalisation, windowing, splits
│
├── reservoir/
│   └── quantum_reservoir.py     ← Ising QRC (PennyLane), qubit sweep, noise
│
├── readout/
│   └── ridge_readout.py         ← Analytical ridge, warm-start, auto-select
│
├── baselines/
│   └── classical.py             ← Persistence, ARIMA, ESN, LSTM/GRU
│
├── experiments/
│   └── sweeps.py                ← Hamiltonian, noise, qubit-scaling, shot sweeps
│
├── evaluation/
│   └── metrics.py               ← RMSE, MAE, VPT, results table
│
├── skill/
│   └── SKILL.md                 ← qBraid agent-executable skill package
│
└── outputs/                     ← All outputs (gitignored except structure)
    ├── raw/                     ← Downloaded NOAA CSVs
    ├── processed/               ← Parsed + resampled Parquet
    ├── results/                 ← CSVs, JSONs, pickled models
    ├── figures/                 ← Plots
    └── logs/                    ← Timestamped run logs
```

---

## Setup (RTX Workstation / MobaXterm)

### 1. Create conda environment

```bash
conda create -n resque python=3.11 -y
conda activate resque
```

### 2. Install PyTorch with CUDA (RTX GPU)

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 \
    -c pytorch -c nvidia -y
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify installation

```bash
python -c "import pennylane; print('PennyLane', pennylane.__version__)"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Running the Pipeline

### Full run (recommended first time)
```bash
python main.py
```

### Smoke test (fast validation, ~5 minutes)
```bash
python main.py --smoke_test
```

### Skip download if data already cached
```bash
python main.py --skip_download
```

### Single horizon only
```bash
python main.py --horizon 6
```

### Skip sweeps (use defaults or previously saved sweep results)
```bash
python main.py --skip_sweeps
```

### On qBraid Lab (no changes needed)
```bash
python main.py --platform qbraid
```

---

## Execution Order (what main.py does)

```
1. Download NOAA ISD CSVs (2018–2024) for Addis Ababa Bole, Ethiopia
2. Parse ISD format → clean DataFrame (T, RH, P, wind speed)
3. Resample to 6h, normalise, build windowed datasets
4. Run parameter sweeps:
   a. Hamiltonian sweep (J, h) → select (J*, h*)
   b. Noise sweep (depolarizing p) → optimal p*
   c. Qubit scaling study (n = 5, 7, 9, 12, 16, 20)
   d. Shot budget ablation
   e. Topology comparison (chain vs all-to-all)
5. Train classical baselines:
   Persistence → ARIMA → ESN → LSTM → GRU
6. Train QRC (cold-start and warm-start, auto-select readout strategy)
7. Evaluate: RMSE, MAE, VPT, results table
8. Save all outputs to outputs/results/
```

---

## Key Config Knobs (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `QUBIT_PRIMARY` | 9 | Primary qubit count |
| `QUBIT_COUNTS` | [5,7,9,12,16,20] | Scaling study range |
| `J_DEFAULT` | 1.0 | Ising coupling strength |
| `H_DEFAULT` | 0.5 | Transverse field strength |
| `NOISE_RATES` | [0,0.005,...,0.05] | Depolarizing noise sweep |
| `WINDOW_SIZE` | 20 | Input window (steps) |
| `HORIZONS` | [6, 24] | Forecast horizons (hours) |
| `USE_WARM_START` | True | ESN → QRC readout init |
| `MULTIOUTPUT_MODES` | all three | Joint/independent/ensemble |

---

## Outputs

All results saved to `outputs/results/`:

| File | Contents |
|---|---|
| `hamiltonian_sweep.csv` | RMSE for each (J, h) pair |
| `best_hamiltonian.json` | Selected J*, h* |
| `noise_sweep.csv` | RMSE per noise rate |
| `qubit_scaling.csv` | RMSE vs qubit count |
| `shot_ablation.csv` | RMSE vs shot budget |
| `readout_selection.json` | Auto-selected readout strategy |
| `results_h6.csv` | Full benchmark table (6h horizon) |
| `results_h24.csv` | Full benchmark table (24h horizon) |
| `warm_start_qrc_config.json` | Reproducibility config |

---

## Reproducibility

All random seeds controlled via `config.RANDOM_SEED = 42`.  
All results reproducible by re-running `python main.py --skip_download`.  
For qBraid agent reproducibility, see `skill/SKILL.md`.

---

## Citation / Attribution

Built for GIC 2026 — Dynamic Systems Forecasting Track B.  
Key references: Hou et al. 2026, Čindrak et al. 2026, Zhu et al. 2025,  
Antoncich et al. 2026, Kornjača et al. 2024.  
AI assistance (Claude, Anthropic) used for code scaffolding and writing — disclosed per GIC rules.