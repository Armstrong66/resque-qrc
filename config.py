"""
config.py — ResQue QRC · Single source of truth for all hyperparameters.
Edit here; every module reads from this. No magic numbers elsewhere.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
DATA_RAW  = ROOT / "outputs" / "raw"
DATA_PROC = ROOT / "outputs" / "processed"
RESULTS   = ROOT / "outputs" / "results"
FIGURES   = ROOT / "outputs" / "figures"
LOGS      = ROOT / "outputs" / "logs"

# ── Dataset ───────────────────────────────────────────────────────────────────
# PRIMARY STATION — change STATION_ID + STATION_NAME to switch location.
# African stations (NOAA ISD, same URL pattern, no API key):
#   Addis Ababa Bole, Ethiopia  → "63450099999"  (Horn of Africa, MAM/OND rains) ← ACTIVE
#   Nairobi Wilson, Kenya       → "63740099999"  (East Africa)
#   Dakar Yoff, Senegal         → "61660099999"  (Sahel / West African monsoon)
# North American reference (data-rich, use for sanity checks):
#   Chicago O'Hare, USA         → "72530094846"
STATION_ID   = "63450099999"          # Addis Ababa Bole (NOAA ISD composite id)
STATION_NAME = "AddisAbaba_Bole"
YEARS        = list(range(2018, 2025)) # 2018–2024 inclusive
NOAA_URL     = "https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{station}.csv"

# Target variables extracted from ISD CSV (column names after parsing)
TARGETS = ["temperature", "humidity", "pressure", "wind_speed"]

# Forecast horizons (in hours, given hourly data resampled to 6h)
HORIZONS = [6, 24]   # 6-hour and 24-hour ahead

# ── Preprocessing ─────────────────────────────────────────────────────────────
RESAMPLE_FREQ   = "6h"          # Resample hourly → 6-hourly
WINDOW_SIZE     = 20            # Input window length (L = 20 steps = 5 days at 6h)
TRAIN_FRAC      = 0.70
VAL_FRAC        = 0.15
TEST_FRAC       = 0.15          # Remainder; no look-ahead leakage enforced
MAX_INTERP_GAP  = 2             # Max consecutive NaN steps to interpolate
# Bump when parser/preprocessing logic changes (invalidates cached Parquet)
DATA_CACHE_VERSION = "v1"
# Bump when windowing/split logic changes (invalidates cached dataset pkl)
PREPROCESS_VERSION = "v1"

# Shared PCA (train-only): same projection for QRC, ESN, LSTM, GRU (fair comparison)
USE_SHARED_PCA     = True
PCA_COMPONENTS     = None         # None -> QUBIT_PRIMARY

# Reservoir transients — keep consistent across fit / predict / sweeps
ESN_WARMUP       = 50
QRC_WARMUP       = 20

# ── Quantum Reservoir ─────────────────────────────────────────────────────────
# Qubit sweep: challenge requires demonstrating performance across 5–20 qubits
QUBIT_COUNTS    = [5, 7, 9, 12, 16, 20]
QUBIT_PRIMARY   = 9             # Primary reported result (matches Hou et al. 2026)

USE_DATA_REUPLOADING = True

# Hamiltonian: transverse-field Ising  H = -J ΣZᵢZⱼ - h ΣXᵢ
J_SWEEP  = [0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
H_SWEEP  = [0.1, 0.3, 0.5, 0.8, 1.0]
J_DEFAULT = 1.0
H_DEFAULT = 0.5

# Topologies to compare
TOPOLOGIES = ["chain", "all_to_all"]
TOPOLOGY_PRIMARY = "chain"      # Overridden by sweep best_topology.json when present

# Time evolution: Trotter steps
TROTTER_STEPS   = 4
EVOLUTION_TIME  = 1.0           # τ per input step

# Measurement: expectation values of σᶻ and σˣ per qubit → 2n features
OBSERVABLES     = ["Z", "X"]    # Both measured; concatenated as readout input

# Feedback memory: re-inject ⟨σᵢᶻ⟩ from t-1 into input at t (Zhu et al. 2025)
USE_FEEDBACK     = True
FEEDBACK_QUBITS  = "all"        # "all" or integer count

# ── Noise sweep (dissipation-assisted QRC) ────────────────────────────────────
NOISE_RATES      = [0.0, 0.005, 0.01, 0.02, 0.05]
NOISE_TYPE       = "depolarizing"   # Applied via PennyLane noise model

# Finite-shot ablation (set to None for exact / infinite shots in simulation)
SHOT_COUNTS      = [None, 500, 1000, 5000]

# Calibration sweeps use fixed contiguous chronological prefixes. This keeps
# reservoir dynamics intact and gives every configuration identical inputs.
# Final reported forecasts still use complete chronological splits.
SWEEP_MAX_TRAIN_SAMPLES = 800
SWEEP_MAX_VAL_SAMPLES   = 200

# Noise sweep: use fewer steps because default.mixed (density matrix) is slower
# than default.qubit (state vector). 50 is enough to select p*; full convergence
# is not required for noise-rate selection.
NOISE_SWEEP_MAX_SAMPLES     = 50
NOISE_SWEEP_MAX_VAL_SAMPLES = 50

# A density-matrix noise sweep is a robustness ablation, not a free primary
# benchmark hyperparameter. Keeping this False prevents a pilot p* from being
# silently propagated into computationally prohibitive scaling/final runs.
USE_SELECTED_NOISE_FOR_PRIMARY = False

# Real-hardware validation (scripts/hardware_validation.py): deliberately
# small — the sweeps above already selected J*/h*/p*/topology*/n_qubits on
# simulator; real QPU time is spent validating ONLY that final config over a
# short subsampled window, not re-running the full pipeline (see
# docs/PROJECT_CRITIQUE.md §3.2 for why running the whole sweep on hardware
# is not feasible).
HARDWARE_VALIDATION_STEPS = 50

# ── QuEra Aquila (analog Rydberg) — reservoir/aquila_backend.py ────────────────
# Aquila has no gate set: our J/h parameters must be re-expressed as REAL
# physical quantities (interaction strength in rad/us via atom spacing, Rabi
# frequency in rad/us) and validated against Aquila's published hardware
# limits (bundled in the bloqade-analog SDK — see get_capabilities() below).
# These scale factors are a DOCUMENTED, CONFIGURABLE DESIGN CHOICE, not a
# literal unit conversion — see docs/PROJECT_CRITIQUE.md for the reasoning
# and what "blockade regime" means here. Tune AQUILA_J_SCALE/AQUILA_H_SCALE
# empirically once real or emulated hardware results are in hand.
AQUILA_J_SCALE   = 20.0   # rad/us of Rydberg interaction V per unit of dimensionless J
AQUILA_H_SCALE   = 3.0    # rad/us of global Rabi frequency Omega per unit of dimensionless h
AQUILA_ENCODE_US = 0.1    # duration of each per-atom local-detuning encoding pulse
AQUILA_DT_US     = 0.3    # duration of each Trotter "evolve" segment (global Rabi drive)
AQUILA_ROTATION_US = 0.05 # duration of the basis-rotation pulse used for the X-readout
# Hardware geometry safety margins (capabilities report exact minimums; a
# small margin avoids landing exactly on a validator boundary due to
# position_resolution rounding).
AQUILA_MIN_SPACING_UM = 4.05
AQUILA_MAX_WIDTH_UM   = 75.0
AQUILA_MAX_HEIGHT_UM  = 76.0
# "local_emulator" (free, Bloqade's own Python simulator — default, safe) |
# "braket_local_emulator" (free, AWS Braket's local AHS simulator — stricter
# validation, closer to what real submission requires) | "aquila" (REAL
# hardware, via AWS Braket — consumes real QPU credits).
AQUILA_SUBMIT_TARGET = "local_emulator"

# ── Readout ───────────────────────────────────────────────────────────────────
# Ridge regression: W* = (XᵀX + λI)⁻¹Xᵀy
RIDGE_LAMBDAS    = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]

# Multi-output strategy: "joint" trains one W* mapping to all targets together;
# "independent" trains one per target; "ensemble" averages independent models.
# The pipeline auto-selects best by val RMSE.
MULTIOUTPUT_MODES = ["joint", "independent", "ensemble"]

# ── Warm-start ────────────────────────────────────────────────────────────────
# The chosen classical model is trained first; its hidden-state features
# initialise the QRC ridge regression via SVD transfer (see
# readout.ridge_readout.classical_warm_start_weights). Switch by editing this
# ONE string — no other code changes needed.
#   "esn"  — EchoStateNetwork.get_reservoir_states()      (recommended default:
#             genuine cross-sample memory in every config, incl. shared PCA)
#   "lstm" / "gru" — RNNWarmStartExtractor.get_hidden_states(); LSTM/GRU are
#             now trained as genuine streaming sequence models (one
#             continuous ordered pass, persistent hidden state, truncated
#             BPTT — see RNN_WARMUP below), so this also carries real
#             cross-sample memory, same as ESN.
#   "arima" is NOT a valid value: ARIMA produces scalar per-target forecasts,
#             not a reservoir-like hidden-state matrix, so it cannot warm-start
#             a ridge readout. Selecting it raises a clear ValueError.
USE_WARM_START     = True
WARM_START_SOURCE  = "esn"      # "esn" | "lstm" | "gru"
ESN_HIDDEN_DIM     = 128        # Matched to QRC feature dimension (2 * n_qubits)

# ── Classical baselines ───────────────────────────────────────────────────────
BASELINES = ["persistence", "arima", "esn", "lstm", "gru"]
# Recurrent model config (LSTM/GRU share this, swappable via flag).
# LSTM/GRU are trained as ONE continuous ordered stream over the whole
# training sequence (batch=1), exactly like EchoStateNetwork/IsingQRC, not as
# independent windowed samples — this is what gives them genuine cross-sample
# memory instead of degrading into a feedforward transform under shared PCA
# (see docs/PROJECT_CRITIQUE.md §3.1, "Pass 4" entry).
RNN_WARMUP      = 50    # Washout before scoring/loss — same role as ESN_WARMUP
# RNN_BATCH is now the truncated-BPTT chunk length (contiguous ordered steps
# per gradient update), not an i.i.d. minibatch size — there is no batch
# dimension left to shuffle once the model sees one continuous sequence.
RNN_HIDDEN      = 64
RNN_LAYERS      = 2
RNN_EPOCHS      = 50
RNN_LR          = 1e-3
RNN_BATCH       = 32

# ── Evaluation metrics ────────────────────────────────────────────────────────
# Weather targets → regression metrics only
# MNIST → accuracy only (classification, separate benchmark)
METRICS_REGRESSION   = ["rmse", "mae", "vpt"]
# VPT: step size = resample grid (6h); Lyapunov time ~2 days for tropical/subseasonal
RESAMPLE_HOURS       = 6
LYAPUNOV_TIME_HOURS  = 48
VPT_THRESHOLD        = 0.4
REPORT_PHYSICAL_UNITS = True

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42
