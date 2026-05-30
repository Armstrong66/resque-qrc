# ResQue-QRC: Project Overview, Critique & Status

**Last updated:** May 30, 2026 — full polish pass (methodology, parity, metrics, figures, tests).

---

## 1. What This Project Is

**ResQue** forecasts multi-output weather at Addis Ababa Bole (`63450099999`) for **GIC 2026 Track B**, using:

- **Quantum reservoir:** transverse-field Ising (PennyLane)
- **Readout:** analytical ridge + ESN warm-start
- **Baselines:** persistence, ARIMA, ESN, LSTM, GRU
- **Sweeps:** Hamiltonian, noise, qubits, shots, topology

```
NOAA ISD -> parse (NaN-safe) -> 6h grid
    -> segment-wise windows + train-only z-score
    -> optional shared PCA -> per-horizon train/eval
    -> results_h6.csv, results_h24.csv + figures/
```

---

## 2. Fix History (All Critical + Deferred Items Addressed)

### Pass 1 — Critical correctness
| Item | Status |
|------|--------|
| Parser NaN / ffill / bfill / dropna | Fixed (`DATA_CACHE_VERSION`) |
| Per-horizon training & results | Fixed |
| Shot ablation wired to PennyLane | Fixed |
| ESN warmup + `label_offset` metrics | Fixed |
| QRC PCA (no `np.resize`) | Fixed |
| Sweep warmup consistency | Fixed |
| Topology from `best_topology.json` | Fixed |

### Pass 2 — Methodology & polish (this pass)
| Item | Status |
|------|--------|
| **Double temporal split** | Fixed — windows built **inside** train/val/test segments only |
| **Shared PCA for all models** | Fixed — `USE_SHARED_PCA`; ESN/LSTM/GRU/QRC share train-only PCA |
| **VPT definition** | Fixed — uses `RESAMPLE_HOURS` (6h), not forecast horizon per row |
| **Physical-unit metrics** | Fixed — `test_rmse_mean_phys` in results when `REPORT_PHYSICAL_UNITS` |
| **`--platform qbraid`** | Fixed — tries `lightning.qubit`, falls back to `default.qubit` |
| **Figures** | Fixed — `evaluation/figures.py` + auto-run after pipeline |
| **`MULTIOUTPUT_MODES`** | Fixed — wired in `RidgeReadout` |
| **Ensemble duplication** | Fixed — ensemble built once post-loop |
| **Warm-start projection** | Improved — SVD transfer instead of random Gaussian |
| **Hamiltonian sweep cost** | Fixed — `SWEEP_MAX_TRAIN_SAMPLES` subsample |
| **Smoke test** | Fixed — auto `--skip_sweeps` |
| **Unit tests** | Added — `tests/test_pipeline.py` |
| **MNIST dead config** | Removed from `config.py` |
| **Dataset cache** | Versioned — `dataset_h{h}_v2.pkl` (`PREPROCESS_VERSION`) |
| **EDA** | Updated for shared PCA + new paths |

---

## 3. What Works Well

- End-to-end reproducible pipeline with `config.py` as single source of truth
- NaN-safe data path with versioned Parquet invalidation
- Leakage-aware preprocessing (train scaler + segment windows)
- Fair model comparison via shared PCA when enabled
- Per-horizon artifacts under `outputs/results/h6/`, `h24/`
- Pre-flight EDA: `python eda/inspect_data.py`
- Sweep plots in `outputs/figures/`
- Five unit tests covering parser, splits, PCA, VPT, alignment

---

## 4. Optional / Future Enhancements (Not Blocking)

| Item | Notes |
|------|--------|
| **Nested validation** | Separate holdout for sweep vs readout selection |
| **CI workflow** | GitHub Action running `pytest` + smoke `main.py` |
| **MNIST expressivity bench** | Out of weather scope; add separate module if required |
| **Iterated multi-step forecasts** | Current setup is direct one-step on 6h grid |
| **Regional Lyapunov calibration** | `LYAPUNOV_TIME_HOURS=48` is documented; tune per station |
| **Full integration tests** | With PennyLane in CI GPU runner |

---

## 5. Configuration Highlights

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `USE_SHARED_PCA` | `True` | Same PCA for QRC, ESN, LSTM, GRU |
| `PREPROCESS_VERSION` | `v2` | Segment-wise windowing |
| `DATA_CACHE_VERSION` | `v2` | NaN-safe parser |
| `ESN_WARMUP` / `QRC_WARMUP` | 50 / 20 | Reservoir transients |
| `SWEEP_MAX_TRAIN_SAMPLES` | 800 | Faster Hamiltonian grid |
| `LYAPUNOV_TIME_HOURS` | 48 | VPT for tropical/subseasonal |
| `REPORT_PHYSICAL_UNITS` | `True` | Extra RMSE columns in results CSV |

---

## 6. Recommended Workflow

```bash
# 1. Rebuild data + datasets after upgrades
python main.py --skip_download --force_rebuild_data --skip_sweeps --skip_baselines --horizon 6

# 2. Audit
python eda/inspect_data.py

# 3. Tests
python -m pytest tests/ -q

# 4. Full run
python main.py

# 5. Fast smoke
python main.py --smoke_test
```

---

## 7. Summary Table

| Area | Status |
|------|--------|
| Data / parser | Complete |
| Preprocessing / splits | Complete |
| Model input parity | Complete (shared PCA) |
| Per-horizon evaluation | Complete |
| QRC / shots / sweeps | Complete |
| Metrics (norm + physical + VPT) | Complete |
| Figures | Complete |
| Tests | Basic suite present |
| qBraid platform hook | Complete (lightning fallback) |
| Documentation | This file + README |

---

## 8. Bottom Line

The project is **methodologically consistent and ready for GIC experiments** after both fix passes. Rebuild processed data and datasets once (`--force_rebuild_data`), run EDA and pytest, then execute the full pipeline. Remaining items in Section 4 are **nice-to-have** research extensions, not blockers for accurate benchmarking.
