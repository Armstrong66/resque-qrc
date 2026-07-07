# ResQue: Warm-Start QRC for Multi-Output Weather Forecasting over East Africa

**Team:** ResQue | **Track:** B — Weather Time-Series Forecasting | **Challenge:** GIC 2026 / qBraid · MITRE · JonesTrading

[![Launch on qBraid](https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png)](https://account.qbraid.com?gitHubUrl=https://github.com/Armstrong66/resque-qrc)

> **For judges:** Open `run_qbraid.ipynb` and run cells sequentially. All results are written to `outputs/results/`. Expected runtime: ~2–4 hours for full run on qBraid GPU instance; ~10 min for smoke test.

---

## What this project does

ResQue applies Quantum Reservoir Computing (QRC) to jointly forecast four atmospheric variables — temperature, humidity, pressure, and wind speed — at 6-hour and 24-hour horizons over East Africa, using NOAA ISD station data from Addis Ababa Bole International Airport (2018–2024). Novel contributions:

1. **Warm-start readout transfer**: ESN readout weights projected via truncated SVD to initialise the QRC ridge readout, reducing sample complexity.
2. **Data reuploading encoding**: Re-encoding input at every Trotter step (Pérez-Salinas et al. 2020) to increase reservoir expressivity without adding qubits.
3. **Calibrated noise characterisation**: Systematic depolarizing noise sweep testing whether NISQ hardware noise assists generalisation (Antoncich et al. 2026).

---

## Project structure

```
resque_qrc/
├── run_qbraid.ipynb          ← ENTRY POINT for judges on qBraid
├── main.py                   ← CLI entry point (local / nohup)
├── run.sh                    ← nohup launcher for local runs
├── config.py                 ← All hyperparameters (edit here only)
├── requirements.txt
├── README.md
│
├── data/
│   ├── downloader.py         ← NOAA ISD CSV download (no API key)
│   └── parser.py             ← ISD parse → clean DataFrame
├── preprocessing/
│   └── pipeline.py           ← PCA + normalise + window + split
├── reservoir/
│   └── quantum_reservoir.py  ← Ising QRC, data reuploading, feedback
├── readout/
│   └── ridge_readout.py      ← Analytical ridge, warm-start SVD, auto-select
├── baselines/
│   └── classical.py          ← Persistence, ARIMA, ESN, LSTM, GRU
├── experiments/
│   └── sweeps.py             ← Hamiltonian, noise, qubit scaling, shots
├── evaluation/
│   └── metrics.py            ← RMSE, MAE, VPT, results table
├── eda/
│   └── inspect_data.py       ← Data quality audit (run before training)
├── skill/
│   └── SKILL.md              ← qBraid agent-executable skill package
│
└── outputs/
    ├── raw/                  ← NOAA ISD CSVs (auto-downloaded)
    ├── processed/            ← Parquet + windowed dataset pickles
    ├── results/              ← All benchmark CSVs, JSONs, model configs
    └── logs/                 ← Timestamped run logs
```

---

## Setup on qBraid Lab

### 1. Open qBraid Lab
Click the **Launch on qBraid** button above, or go to `account.qbraid.com` and open this repo.

### 2. Select kernel
Open `run_qbraid.ipynb` and select the **Python 3 [PennyLane]** kernel (pre-installed on qBraid).

### 3. Install missing dependencies
Cell 1 of the notebook handles this automatically:
```python
# Runs automatically in Cell 1:
pip install pmdarima pyarrow pennylane-lightning -q
```

### 4. Run all cells sequentially
That's it. No external configuration required.

---

## Setup locally (RTX workstation / MobaXterm)

```bash
# 1. Create conda environment
conda create -n resque-qrc python=3.11 -y
conda activate resque-qrc

# 2. PyTorch with CUDA (do before pip)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 \
    -c pytorch -c nvidia -y

# 3. Remaining dependencies
pip install -r requirements.txt

# 4. Verify
python -c "import pennylane; import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Step-by-step: reproducing results

### Option A — Notebook (recommended for judges)
```
1. Open run_qbraid.ipynb
2. Run Cell 1  — environment check
3. Run Cell 2  — set RUN_MODE = 'full' for full results, 'smoke' for quick test
4. Run Cell 3  — download NOAA ISD data
5. Run Cell 4  — parse and inspect (confirm 0% NaN in all targets)
6. Run Cell 5  — preprocess (PCA + windowing + splits)
7. Run Cell 6  — classical baselines (Persistence, ARIMA, ESN, LSTM, GRU)
8. Run Cell 7  — encoding ablation (standard vs. data reuploading)
9. Run Cell 8  — QRC training (cold-start and warm-start)
10. Run Cell 9  — qubit scaling study (n = 5, 7, 9, 12, 16, 20)
11. Run Cell 10 — full benchmark tables (6h and 24h)
12. Run Cell 11 — noise sweep
13. Run Cell 12 — shot budget ablation
14. Run Cell 13 — hardware run (QuEra Aquila or IBM Eagle, when QPU ready)
15. Run Cell 14 — output summary and Phase 3 checklist
```

### Option B — CLI (full run, background)
```bash
bash run.sh                          # Full run, backgrounded
bash run.sh --smoke_test             # Fast test (~10 min)
bash run.sh --skip_download          # Re-run with cached data
tail -f outputs/logs/run_*.log       # Monitor live
```

### Option C — Single command
```bash
python main.py                       # Full pipeline
python main.py --smoke_test          # Smoke test
```

---

## Expected inputs and outputs

| Input | Source |
|---|---|
| NOAA ISD CSVs | Auto-downloaded from `ncei.noaa.gov/data/global-hourly/access/{year}/63450099999.csv` |
| No API key | Fully public, no authentication |

| Output file | Contents |
|---|---|
| `outputs/results/results_h6.csv` | Full benchmark table, 6h horizon |
| `outputs/results/results_h24.csv` | Full benchmark table, 24h horizon |
| `outputs/results/qubit_scaling.csv` | RMSE vs qubit count (n=5→20) |
| `outputs/results/noise_sweep.csv` | RMSE vs depolarizing noise rate |
| `outputs/results/shot_ablation.csv` | RMSE vs shot budget |
| `outputs/results/encoding_ablation.json` | Standard vs. data reuploading RMSE |
| `outputs/results/hamiltonian_sweep.csv` | (J, h) sweep RMSE grid |
| `outputs/results/warm_start_qrc_config.json` | Full reproducibility config |
| `outputs/results/readout_selection.json` | Auto-selected readout strategy log |

---

## Key configuration parameters (`config.py`)

| Parameter | Value | Description |
|---|---|---|
| `STATION_ID` | `63450099999` | Addis Ababa Bole, Ethiopia |
| `YEARS` | 2018–2024 | 7 years of hourly observations |
| `QUBIT_PRIMARY` | 9 | Primary reported result |
| `QUBIT_COUNTS` | [5,7,9,12,16,20] | Scaling study range |
| `J_DEFAULT` | 0.3 | Ising coupling (from sweep) |
| `H_DEFAULT` | 0.8 | Transverse field (from sweep) |
| `TOPOLOGY_PRIMARY` | `chain` | Best topology from sweep |
| `WINDOW_SIZE` | 20 | Input window (5 days at 6h) |
| `HORIZONS` | [6, 24] | Forecast horizons (hours) |
| `USE_WARM_START` | True | ESN → QRC SVD init |
| `TROTTER_STEPS` | 4 | Trotter steps per input |

---

## Hardware access (Phase 3)

| Backend | How to access on qBraid |
|---|---|
| QuEra Aquila (primary) | qBraid Bloqade kernel → `github.com/QuEraComputing/QRC-tutorials` |
| IBM Eagle/Heron (fallback) | qBraid IBM token → Cell 13 of notebook |
| PennyLane lightning.qubit (GPU sim) | Pre-installed on qBraid; auto-selected for n ≥ 9 |

---

## Known limitations

- **QRC vs ESN gap at pilot scale**: Cold/warm-start QRC at 1.212 normalised RMSE vs ESN at 0.685 in pilot. Gap attributed to single-injection encoding bottleneck — data reuploading is the primary Phase 3 fix.
- **Pressure channel missingness**: Addis Ababa ISD has intermittent SLP gaps. Per-variable independent readout with horizon-specific λ applied as mitigation.
- **Warm-start at pilot scale**: Cold and warm QRC tied in pilot — SVD transfer benefit expected to widen on full dataset where ESN has more training signal to transfer.
- **Hardware noise**: Noise sweep showed p*=0 optimal in simulation. Whether hardware-induced Aquila noise regularises generalisation remains the key open hardware question.
- **AI disclosure**: Claude (Anthropic) used for code scaffolding and write-up assistance. Technical contributions, formulations, and results are the team's own work. Disclosed per GIC rules.

---

## Citation

```bibtex
@misc{resque2026,
  title   = {ResQue: Warm-Start Quantum Reservoir Computing for Multi-Output Weather Forecasting},
  authors = {Joseph Derrick Anane Nti Koduah, Elliot Amponsah, Godfred Addo Boakye, Calvin Sewornu, Emmanuel Omolajah},
  year    = {2026},
  note    = {GIC 2026, Track B (Weather Forecasting), qBraid/MITRE/JonesTrading}
}
```

**Key references:** Kornjača et al. (2024) arXiv:2407.02553 · Zhu et al. (2025) arXiv:2405.04799 · Čindrak et al. (2026) arXiv:2603.21371 · Antoncich et al. (2026) arXiv:2602.14641 · Pérez-Salinas et al. (2020) Quantum 4:226
