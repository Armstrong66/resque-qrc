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
STATION_ID   = "63450099999"          # Addis Ababa Bole: USAF=634010 + WBAN=99999 (Horn of Africa, MAM/OND seasonal cycle)
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

# ── Quantum Reservoir ─────────────────────────────────────────────────────────
# Qubit sweep: challenge requires demonstrating performance across 5–20 qubits
QUBIT_COUNTS    = [5, 7, 9, 12, 16, 20]
QUBIT_PRIMARY   = 9             # Primary reported result (matches Hou et al. 2026)

# Hamiltonian: transverse-field Ising  H = -J ΣZᵢZⱼ - h ΣXᵢ
J_SWEEP  = [0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
H_SWEEP  = [0.1, 0.3, 0.5, 0.8, 1.0]
J_DEFAULT = 1.0
H_DEFAULT = 0.5

# Topologies to compare
TOPOLOGIES = ["chain", "all_to_all"]
TOPOLOGY_PRIMARY = "chain"      # Default; all_to_all run as ablation

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

# ── Readout ───────────────────────────────────────────────────────────────────
# Ridge regression: W* = (XᵀX + λI)⁻¹Xᵀy
RIDGE_LAMBDAS    = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]

# Multi-output strategy: "joint" trains one W* mapping to all targets together;
# "independent" trains one per target; "ensemble" averages independent models.
# The pipeline auto-selects best by val RMSE.
MULTIOUTPUT_MODES = ["joint", "independent", "ensemble"]

# ── Warm-start ────────────────────────────────────────────────────────────────
# ESN trained first; its W_out initialises QRC ridge regression
USE_WARM_START   = True
ESN_HIDDEN_DIM   = 128          # Matched to QRC feature dimension (2 * n_qubits)

# ── Classical baselines ───────────────────────────────────────────────────────
BASELINES = ["persistence", "arima", "esn", "lstm", "gru"]
# Recurrent model config (LSTM/GRU share this, swappable via flag)
RNN_HIDDEN      = 64
RNN_LAYERS      = 2
RNN_EPOCHS      = 50
RNN_LR          = 1e-3
RNN_BATCH       = 32

# ── MNIST expressivity benchmark ─────────────────────────────────────────────
MNIST_N_TRAIN   = 1000          # Subset for speed; full 60k not needed here
MNIST_N_TEST    = 200
MNIST_N_CLASSES = 10

# ── Evaluation metrics ────────────────────────────────────────────────────────
# Weather targets → regression metrics only
# MNIST → accuracy only (classification, separate benchmark)
METRICS_REGRESSION   = ["rmse", "mae", "vpt"]
LYAPUNOV_TIME_HOURS  = 84       # ~3.5 days; midlatitude synoptic Lyapunov time
VPT_THRESHOLD        = 0.4      # Normalised error threshold for VPT

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42