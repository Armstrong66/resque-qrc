# ResQue: Quantum Reservoir Computing for Weather Forecasting

ResQue is a reproducible research pipeline for multivariate weather forecasting with a transverse-field Ising quantum reservoir computer (QRC). It forecasts temperature, relative humidity, sea-level pressure, and wind speed at 6-hour and 24-hour horizons from NOAA Integrated Surface Database observations at Addis Ababa Bole, Ethiopia, covering 2018--2024.

The pipeline uses strictly chronological data splits, train-only preprocessing, classical baselines, bounded simulator calibration studies, and separately scoped hardware validation.

## Highlights

- Trotterized transverse-field Ising QRC with data reuploading and output feedback.
- Leakage-safe chronological 70/15/15 train, validation, and test partitions.
- Train-only PCA shared by the QRC and learned classical baselines.
- Cold-start and SVD-based warm-start ridge readouts.
- Persistence, ARIMA, ESN, LSTM, and GRU baseline models.
- Reproducible Hamiltonian, topology, qubit, shot, and simulator-noise experiments.
- Optional IBM and QuEra Aquila validation paths after simulator selection.

## Repository layout

```text
main.py                         Primary CLI pipeline
config.py                       Central experiment configuration
run.sh                          Background Linux launcher
data/                           NOAA download and parsing
preprocessing/                  Splits, normalization, PCA, and windows
reservoir/                      Gate-model QRC and Aquila analog backend
readout/                        Ridge readout and warm-start transfer
baselines/                      Persistence, ARIMA, ESN, LSTM, and GRU
experiments/                    Calibration and robustness sweeps
evaluation/                     Metrics, result tables, and figures
scripts/hardware_validation.py  Small hardware/emulator validation utility
docs/METHODS_DRAFT.md           Paper-oriented methodological description
outputs/                        Generated data, logs, figures, and results
```

`run_qbraid.ipynb`, `agent_runner.py`, and `skill/` are retained for the separate qBraid integration workflow; they are not required for the standard CLI pipeline.

## Installation

Python 3.11 is recommended. On a Linux workstation with an NVIDIA GPU:

```bash
conda create -n resque-qrc python=3.11 -y
conda activate resque-qrc
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.txt
python -c "import pennylane, torch; print('CUDA available:', torch.cuda.is_available())"
```

`pmdarima` is preferred for ARIMA order selection but has a statsmodels fallback. Hardware validation requires additional optional dependencies; see below.

## Running the pipeline

Run commands from the repository root.

```bash
# Fast integrity check; uses the first 500 time points and skips sweeps.
python main.py --smoke_test

# Full single-horizon runs.
python main.py --horizon 6
python main.py --horizon 24

# Full two-horizon pipeline.
python main.py

# Reuse downloaded observations or existing sweep selections.
python main.py --skip_download
python main.py --skip_sweeps
```

For a persistent remote Linux session:

```bash
bash run.sh --horizon 6
tail -f outputs/logs/run_*.log
```

Logs report sweep timings, reservoir progress, ARIMA walk-forward progress, and recurrent-model epochs, making long simulations distinguishable from failed processes.

## Experimental protocol

Observations are resampled to a six-hour grid. Each input is a 20-step history (five days) of four variables, flattened to 80 values. PCA is fit on training windows only and reduces the input to the active qubit count (nine by default). Windows are constructed within, never across, temporal partitions.

The QRC implements

$$H = -J \sum_{\langle i,j \rangle} Z_i Z_j - h \sum_i X_i,$$

using four Trotter steps per input sample. The reservoir state contains expectation values of both $Z$ and $X$ observables, producing $2n$ features for $n$ qubits. The readout selects the best validation RMSE among joint, independent, and ensemble ridge strategies. Cold-start and warm-start readouts share the same QRC trajectories. For the warm-start condition, ESN, LSTM, and GRU hidden-state sources are each transferred into the QRC readout and compared by the resulting hybrid validation RMSE; the best available source is selected independently for each horizon.

The primary benchmark uses all available chronological examples in each split. Calibration and robustness experiments instead use a common contiguous prefix for every candidate, preserving reservoir dynamics while bounding cost: up to 800 training and 200 validation windows by default. The density-matrix noise screen uses 50 training and 50 validation windows. These counts are saved with sweep artifacts.

For each horizon, the primary architecture is selected by a paired standard-versus-data-reuploading Hamiltonian grid on identical calibration windows. The selected encoding and its re-optimized $J,h$ values are then used only for that horizon's downstream QRC evaluation. The default primary protocol is noiseless (`p = 0`). A selected depolarizing-noise value is treated as a simulator robustness result and is not silently propagated into the primary forecasts, scaling study, or shot ablation. To run the distinct noisy-QRC protocol explicitly:

```bash
python main.py --use_selected_noise
```

See [the methods draft](docs/METHODS_DRAFT.md) for the complete paper-oriented description of the data, model, baselines, and reporting boundaries.

## Configuration

All settings are centralized in `config.py`.

| Setting | Default | Purpose |
|---|---:|---|
| `HORIZONS` | `[6, 24]` | Forecast horizons in hours |
| `WINDOW_SIZE` | `20` | Six-hourly history length |
| `QUBIT_PRIMARY` | `9` | Primary QRC size |
| `QUBIT_COUNTS` | `[5, 7, 9, 12, 16, 20]` | Qubit-scaling study |
| `ENCODING_OPTIONS` | `[False, True]` | Paired standard/reuploading candidates; standard resolves exact ties |
| `USE_DATA_REUPLOADING` | `True` | Default for standalone QRC construction |
| `USE_FEEDBACK` | `True` | Feed prior $\langle Z\rangle$ into encoding |
| `TROTTER_STEPS` | `4` | Trotter steps per input |
| `WARM_START_SOURCES` | `["esn", "lstm", "gru"]` | Hybrid-QRC warm-start ablation candidates |
| `WARM_START_SOURCE` | `"esn"` | Standalone/legacy default source |
| `SWEEP_MAX_TRAIN_SAMPLES` | `800` | Standard calibration training budget |
| `SWEEP_MAX_VAL_SAMPLES` | `200` | Standard calibration validation budget |
| `USE_SELECTED_NOISE_FOR_PRIMARY` | `False` | Keep noise screen separate by default |

Cached sweep winners record the active data-reuploading setting. The pipeline warns if a cached selection was generated under a different encoding configuration.

## Outputs and verification

Generated artifacts are written below `outputs/`.

| Location | Contents |
|---|---|
| `outputs/raw/` | Downloaded NOAA ISD observations |
| `outputs/processed/` | Cleaned data and serialized horizon datasets |
| `outputs/logs/` | Timestamped run logs |
| `outputs/results/results_h6.csv` | 6-hour benchmark table |
| `outputs/results/results_h24.csv` | 24-hour benchmark table |
| `outputs/results/hamiltonian_sweep_h{h}_{encoding}.csv` | Encoding-specific Hamiltonian grid for horizon `h` |
| `outputs/results/encoding_hamiltonian_comparison_h{h}.csv` | Paired encoding--Hamiltonian comparison for horizon `h` |
| `outputs/results/best_qrc_architecture_h{h}.json` | Horizon-specific selected encoding and Hamiltonian |
| `outputs/results/h{h}/warm_start_source_ablation.json` | Hybrid-QRC validation comparison of ESN, LSTM, and GRU transfer sources |
| `outputs/results/noise_sweep.csv` | Simulator noise-robustness results |
| `outputs/results/qubit_scaling.csv` | Qubit-scaling results |
| `outputs/results/shot_ablation.csv` | Finite-shot robustness results |
| `outputs/results/h{h}/` | Per-horizon models, QRC configuration, and selection logs |

Readout logs are mode-specific: `cold_start_qrc_readout_selection.json` and `warm_start_qrc_readout_selection.json`. Verify a completed run with:

```bash
python verify_results.py
```

## Hardware validation

Hardware backends are reserved for short validation sequences after simulator configuration selection; they are never used for parameter sweeps.

```bash
# Simulator control.
python scripts/hardware_validation.py --horizon 6 --n_steps 50

# QuEra Aquila local emulator; requires bloqade-analog.
QRC_BACKEND=aquila python scripts/hardware_validation.py --horizon 6 --n_steps 10

# IBM backend; requires pennylane-qiskit, qiskit-ibm-runtime, and an account token.
QRC_BACKEND=ibm python scripts/hardware_validation.py --horizon 6 --n_steps 10
```

The Aquila path is an analog Rydberg implementation, not a gate-model device swap. It maps $J$ to atom geometry, $h$ to global Rabi drive, and inputs to local detuning. Its all-to-all topology is a documented compact-ring approximation. Emulator output must not be represented as a live-hardware result.

## Citation

```bibtex
@misc{resque2026,
  title  = {ResQue: Quantum Reservoir Computing for Multivariate Weather Forecasting},
  author = {Joseph Derrick Anane Nti Koduah, Elliot Amponsah, Godfred Addo Boakye, Calvin Sewornu, Emmanuel Omolajah},
  year   = {2026}
}
```
