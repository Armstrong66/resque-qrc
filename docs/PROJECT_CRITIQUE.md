# ResQue-QRC — Project Critique
**Judge's pass · GIC 2026 Track B · reviewed as a $1,000-grant submission**

**Last updated:** 2026-07-18 — Pass 5 (QuEra Aquila — the team's confirmed PRIMARY hardware backend — implemented for real as a physical Rydberg-atom program, validated against two free local emulators; IBM fallback path strengthened with a real device-construction + circuit-execution test against an IBM fake backend)

---

## 0. Verdict, up front

Passes 1–3 established a methodologically careful simulation pipeline and fixed the ARIMA-reliability, warm-start-configurability, and dead-reuploading-flag issues found along the way. Pass 4 fixed the three gaps it had flagged but not yet closed: LSTM/GRU statefulness, the missing agentic tooling (`agent_runner.py`/`verify_results.py`), and — for hardware — implemented IBM while leaving QuEra Aquila as an honest `NotImplementedError` rather than a guess, because at that point the team's actual hardware allocation wasn't confirmed and Aquila's physics genuinely can't be faked safely.

**This pass exists because that allocation is now confirmed**: the team's qBraid credits cover QuEra's Aquila (**primary**) and IBM Eagle (**fallback**). That changes the calculus — Aquila is no longer a hypothetical "maybe later," it's the backend the project is actually going to be judged on. This pass implements it for real:

1. **QuEra Aquila is now a genuine, working execution path** (`reservoir/aquila_backend.py`) — not a device swap (Aquila has no gate set), but a from-scratch translation of the transverse-field Ising Hamiltonian into a real analog Rydberg program: atom spacing realizes J via the Rydberg blockade, a global Rabi drive realizes h, and per-atom local detuning realizes the classical encoding. Built against QuEra's own `bloqade-analog` SDK, whose actual installed API (not memory of how it "probably" works) drove every design decision — see §1.
2. **The IBM fallback path is now verified one level deeper**: beyond confirming imports resolve, this pass actually constructed the `qiskit.remote` PennyLane device against a real IBM fake backend (`FakeSherbrooke`, 127 qubits — simulates real IBM hardware topology) and ran this project's exact gate sequence (RY/IsingZZ/RX) through it successfully — see §2.

Neither backend has been run against **live** hardware — that's stated plainly, not hedged. What changed this pass is the gap between "described in a docstring" and "actually executes correctly," which is now closed for both, using the strongest verification available without live credentials.

---

## 1. QuEra Aquila — implemented, and how

### 1.0 Why this wasn't attempted in Pass 4, and why it changed

Pass 4's refusal to implement Aquila was not caution for its own sake — it was that guessing at atom placement, pulse timing, and unit conversions for a **real, physical, paid** hardware submission is a materially worse outcome than admitting the gap. That reasoning still holds. What changed is that this pass had the actual `bloqade-analog` SDK installed and introspected — its bundled hardware capabilities file, its real builder API, and (critically) its **free local emulators**, which let every design decision below be executed and checked before being written down as fact. Nothing here was written from memory of "how Rydberg physics probably works."

### 1.1 Grounding: what was actually looked up, not assumed

`bloqade-analog` (QuEra's own Python SDK; the metapackage `bloqade` also pulls in a much larger, unrelated gate-based toolchain — `bloqade-analog` alone is the correct, minimal dependency) was installed in an isolated venv and its **bundled capabilities file** (`bloqade/analog/submission/config/capabilities.json` — QuEra's own published Aquila spec, shipped inside the SDK, not guessed) was read directly:

| Limit | Value | Consequence for this implementation |
|---|---|---|
| Max program duration | **4 μs total** | All Trotter/encode segments must fit in one continuous ≤4μs pulse window — verified the default config (`trotter_steps=4`) comfortably fits (~1.6–2.4μs). |
| Lattice area | 75 × 76 μm | Hard ceiling on how many atoms fit in a chain/ring at a given spacing. |
| Minimum atom spacing | 4 μm | Hard floor — closer than this isn't a hardware-realizable geometry. |
| Global Rabi frequency | 0–15.8 rad/μs | Bounds the transverse-field (`h`) scale factor. |
| Local detuning | **0 to 125 rad/μs, per-atom coefficient ∈ [0, 1] — non-negative only** | This is the one that actually shapes the design: there is no way to encode a signed value directly through local addressing (see §1.2). |
| Shots | 1–1000 | Hard cap; `AquilaBackend` clamps and logs if a caller asks for more. |
| `RB_C6` (Rydberg interaction constant) | 5.42×10⁶ rad·μm⁶/μs (`bloqade.RB_C6`) | Used directly, not re-derived, to convert requested `J` into atom spacing. |

### 1.2 The physical mapping, and where it's a design choice vs. a constraint

- **J → atom spacing.** `spacing = (RB_C6 / (J × AQUILA_J_SCALE))^(1/6)`, clamped to fit both the hardware minimum (4 μm) **and** the lattice area for the requested `n_qubits` — bidirectionally, which mattered in practice (see §1.4). The realized J is logged and used in place of the requested one whenever clamping changes it materially.
- **h → global Rabi amplitude**, scaled by `config.AQUILA_H_SCALE`. **`AQUILA_J_SCALE`/`AQUILA_H_SCALE` are a documented design choice**, chosen to land in the blockade-dominant regime (interaction strength ≫ Rabi frequency) that makes the physics approximate an Ising-like model — not a value measured against real hardware. Tune empirically once real or emulated results are in hand.
- **Classical input encoding → per-atom local detuning coefficients.** This is the one genuine hardware **constraint**, not a choice: local detuning site-coefficients are restricted to `[0, 1]` (non-negative) by the hardware itself. Our existing `[-π, π]` encoding angles (already NaN-guarded, already used by the gate-based path — reused here via the same `_encoding_angles` helper, not reimplemented) are affine-remapped: `coefficient = (angle/π + 1) / 2`.
- **Measurement → Z only, natively.** Aquila measures Rydberg-occupation (Z-like) only. There is no native ⟨X⟩. The X-feature half is obtained by appending a short resonant π/2 pulse before measurement (a standard basis-rotation technique) and submitting a **second** program — this doubles shots/cost per timestep relative to the gate-based path, which is the honest price of getting both feature halves from a device with one native measurement basis.
- **"all_to_all" topology is an approximation, stated as one.** 1/r⁶ interactions can't realize uniform coupling beyond nearest neighbors on any real 2D placement. Atoms are placed on a compact ring so *adjacent* atoms hit the target spacing; more distant pairs on the ring couple more weakly. `"chain"` has no such caveat — it's a literal 1D chain.

### 1.3 A bug the stricter validator caught — exactly what local-emulator testing is for

The first working version passed Bloqade's own (permissive) local Python emulator but was rejected by **AWS Braket's local AHS emulator** — the same validator real hardware submission goes through — with a genuine error: the detuning and Rabi-amplitude channels had **mismatched total durations** (one waveform implicitly ended early). This is precisely the class of mistake that would have either failed at real submission time or, worse, been silently accepted with undefined behavior on a laxer validator. Fixed by making every pulse channel explicitly span the full program duration, including flat/zero segments. Re-verified clean against both emulators afterward. This is the concrete reason §2 of Pass 4 (and this pass) insists on testing against the *strictest* available free validator, not just whichever one happens not to complain.

### 1.4 A geometry bug found by testing the project's own real parameter ranges, not toy values

The first version of the geometry function only clamped atom spacing **up** to the hardware minimum and, if the resulting chain didn't fit the lattice area, simply raised an error — it never tried **reducing** spacing (within the hardware-allowed range) to make a larger `n_qubits` fit. Concretely: for `config.QUBIT_COUNTS` = `[5, 7, 9, 12, 16, 20]` and `config.J_SWEEP`, this incorrectly rejected `n_qubits=12` and `16` at *every* J value, when a smaller (but still hardware-valid) spacing would have fit them. Found by testing the actual project parameter ranges, not small toy values, which is exactly why that testing was done. **Fixed**: spacing is now clamped to the tighter of `[hardware minimum, area-limited maximum]`, and the realized J is logged whenever it had to move away from the requested value. Result, checked directly against the real sweep ranges: `n_qubits` up to 16 fit as a chain (with J drifting toward higher values as clamping intensifies); `n_qubits=20` does **not** fit as a chain even at minimum spacing, but **does** fit as `topology="all_to_all"` (the ring approximation packs more efficiently) — a genuinely useful, non-obvious finding for planning which topology to actually submit for the largest qubit count in the scaling study.

### 1.5 Verified

All of the following were executed against the installed SDK, not asserted:
- Full `IsingQRC(hardware_backend="aquila")` → `run_sequence()` round trip, both `use_data_reuploading=True/False`, correct `(T-warmup, 2n)` shapes, zero NaNs, sensible bounded `[-1, 1]` outputs.
- Re-run against **both** free local emulators (`"local_emulator"` and the stricter `"braket_local_emulator"`) — both clean after the §1.3 fix.
- `topology="all_to_all"` end-to-end (with its approximation warning firing correctly).
- Geometry `ValueError` fires correctly and clearly for a genuinely-too-large `n_qubits`, and does **not** fire incorrectly for the project's real `(n_qubits, J)` ranges after the §1.4 fix.
- Encoding sensitivity: two different input vectors produce genuinely different output features (not degenerate/constant noise).
- `config.AQUILA_SUBMIT_TARGET` / `$AQUILA_SUBMIT_TARGET` env-var override confirmed to correctly select between the two free emulators (real-hardware submission itself was not exercised — no AWS Braket credentials in this environment).
- `run_sequence()`'s blanket `if not PENNYLANE_AVAILABLE: raise ImportError` incorrectly fired for the Aquila-only path (which has nothing to do with PennyLane) — found by testing in an environment with `bloqade-analog` but not `pennylane` installed, fixed to only apply outside the Aquila backend.

**What was not, and could not be, verified here:** submission to live Aquila hardware (no AWS Braket credentials in this environment), and whether `AQUILA_J_SCALE`/`AQUILA_H_SCALE` actually land the physics in a useful regime for *this project's specific weather-forecasting task* — that can only be answered by real (or at least real-emulator) benchmark results, which is exactly what running `scripts/hardware_validation.py --backend aquila` is for.

---

## 2. IBM Eagle/Heron — verified one level deeper

Pass 4 confirmed the IBM path's imports resolve correctly and raises a clear error without credentials. This pass went further, without needing real credentials: installed `qiskit-ibm-runtime` and `pennylane-qiskit`, confirmed `"qiskit.remote"` is a genuinely registered PennyLane device (not a guessed string), and — using `qiskit_ibm_runtime.fake_provider.FakeSherbrooke` (a 127-qubit fake backend that mirrors real IBM hardware topology, free, no account needed) — actually **constructed** the device (`qml.device("qiskit.remote", wires=5, backend=FakeSherbrooke(), shots=100)`) and **ran this project's exact circuit structure** (RY encode → IsingZZ couplings → RX field, the same three gate types `_build_qnode` uses) through it successfully, returning bounded, sensible expectation values. This confirms the device-construction pattern in `_get_ibm_device` is correct against a real IBM-provided fake backend, not just plausible-looking code. The one piece still unverified is `QiskitRuntimeService()`'s real account/token handling, which genuinely requires live credentials this environment doesn't have.

---

## 3. What remains genuinely open (honest, not exhaustive)

- **Neither Aquila nor IBM has been run against live hardware.** Both are now implemented and verified as far as possible without credentials — that is a meaningfully stronger claim than Pass 4's, but "verified without live hardware" and "verified on live hardware" are different claims. Run `scripts/hardware_validation.py --backend aquila --n_steps 5` for real, first, before a larger or IBM run.
- **`AQUILA_J_SCALE`/`AQUILA_H_SCALE` need empirical tuning.** They're chosen to be physically sensible (blockade-dominant regime), not fit to this project's data. The first real (or `braket_local_emulator`) run's RMSE is the actual test of whether they're in a useful range.
- **Aquila's "all_to_all" is an approximation** (§1.2) — treat comparisons between `chain` and `all_to_all` hardware results with that in mind; they are not testing the same underlying interaction structure that the simulator's `all_to_all` (literal, uniform ZZ coupling) tests.
- **No reference run is saved for `verify_results.py`'s tolerance check** — unchanged from Pass 4, still correctly deferred until a trusted post-fix full run exists.
- **Everything previously listed as out of scope remains out of scope**: `scripts/export_model.py` / the Vercel inference app, nested/held-out validation for sweep selection, CI. None of these were asked for.

---

## 4. What already works well (credit where due, carried forward and still true)

- **Leakage discipline is genuinely good**: train-only z-score, segment-wise windowing strictly inside train/val/test boundaries, versioned Parquet/pickle caches that force a rebuild rather than silently serving stale data.
- **`config.py` as single source of truth** holds up — every new configurable choice across all five passes (`WARM_START_SOURCE`, `USE_DATA_REUPLOADING`, `RNN_WARMUP`, `HARDWARE_VALIDATION_STEPS`, and now the `AQUILA_*` physical-mapping constants) lives there, none hardcoded elsewhere.
- **The auto-selecting `RidgeReadout`** remains a properly honest piece of model selection.
- **The project now has two real, distinct, independently-verified hardware execution paths** rather than one real path and one aspiration — and the harder of the two (Aquila, genuinely different physics, no gate set) got the more careful treatment precisely because it was harder, not less.

---

## 5. Current pipeline flow (as of this pass)

```
NOAA ISD CSV (download_all)
  → parse_isd_csv (Magnus RH, sentinel→NaN)
  → resample 6h, interpolate short gaps, drop remaining NaN rows
  → versioned Parquet cache (DATA_CACHE_VERSION)
  → WeatherPreprocessor: train-only z-score → per-segment windowing
  → versioned pickle cache (PREPROCESS_VERSION)
  → [if USE_SHARED_PCA] fit PCA on X_train only → project all splits to
    n_qubits dims — same projection used by QRC, ESN, LSTM, GRU

Sweeps (experiments/sweeps.py — simulator ONLY, by design, all encoding-aware):
  hamiltonian_sweep → best_hamiltonian.json {J*, h*, use_data_reuploading}
  noise_sweep       → best_noise.json {p*, use_data_reuploading}
  topology_comparison → best_topology.json {topology*, use_data_reuploading}
  qubit_scaling_study / shot_ablation (noise+shots now composable)
  [staleness warning if a cached JSON's recorded encoding != current config]

Per horizon (main.py::_train_horizon / agent_runner.py train_baselines+train_qrc):
  Baselines → persistence, arima (2-backend), esn, lstm/gru (genuinely
    stateful streaming models) → baseline_status.json per horizon
  Warm-start source resolution (WARM_START_SOURCE: esn|lstm|gru, ARIMA rejected)
  QRC (IsingQRC): cold_start_qrc / warm_start_qrc, under J*/h*/p*/topology*/
    USE_DATA_REUPLOADING, hardware_backend="simulation" by default
  RidgeReadout.fit → auto-select joint/independent/ensemble × λ
  → results_h{6,24}.csv; QRC predictions persisted to disk (not just
    readout weights), so benchmark_all can reconstruct standalone

Real-hardware validation (scripts/hardware_validation.py — separate from the
  above, deliberately NOT part of the main sweep/train flow):
  Load the FINAL selected {mode}_config.json + {mode}_readout.pkl
  → small subsampled test window → run on simulation (always, paired
    baseline) + requested QRC_BACKEND:
      "aquila" (PRIMARY) → reservoir/aquila_backend.py::AquilaBackend
        (real physical Rydberg program; AQUILA_SUBMIT_TARGET picks free
        emulator vs. real hardware)
      "ibm" (FALLBACK) → pennylane-qiskit device swap, existing gate circuit
  → hardware_validation.json with the paired sim-vs-hardware comparison

Agentic layer (agent_runner.py / verify_results.py — same underlying
  functions as above, task-per-process instead of one monolithic run):
  setup → download_data → preprocess → eda → hamiltonian_sweep → noise_sweep
  → topology_sweep → encode_ablation → train_baselines → qubit_scaling
  → shot_ablation → train_qrc → benchmark_all → verify_results  (full_run)

Figures (evaluation/figures.py) → outputs/figures/*.png
```

---

## 6. Summary table (cumulative across all passes)

| Area | Status |
|---|---|
| ARIMA baseline reliability | **Fixed** (Pass 3) |
| Warm-start source configurability | **Fixed** (Pass 3) |
| Data reuploading wired end-to-end | **Fixed** (Pass 3) |
| Noise+shots device interaction | **Fixed** (Pass 3) |
| ESN short-sequence crash + RMSE correctness | **Fixed** (Pass 3 + Pass 4) |
| `encoding_ablation` ported to real module | **Fixed** (Pass 3) |
| LSTM/GRU cross-sample memory under shared PCA | **Fixed** (Pass 4) — genuinely stateful streaming models |
| `agent_runner.py` / `verify_results.py` | **Built and tested** (Pass 4) |
| QRC predictions not persisted (agentic contract gap) | **Fixed** (Pass 4) |
| PennyLane `set_shots` deprecation | **Fixed** (Pass 4) |
| torch/numpy ABI risk | **Detected and reported at `setup` time** (Pass 4) |
| **QuEra Aquila hardware path** | **Implemented for real** (Pass 5, §1) — physical Rydberg program, validated against two free local emulators, two real bugs found and fixed via that validation, not yet run on live hardware |
| **IBM hardware path** | **Implemented + verified against a real IBM fake backend** (Pass 5, §2) — device construction and this project's exact gate sequence confirmed working, not yet run on live hardware |
| Tests | 5/5 passing throughout every pass |

---

## 7. Bottom line

The team's qBraid hardware allocation is QuEra Aquila (primary) and IBM Eagle (fallback) — that was the missing piece of information that made a real Aquila implementation possible and worthwhile rather than speculative. It is now real: a physical Rydberg-atom program that reuses this project's existing encoding/warmup/feedback logic, respects Aquila's actual published hardware limits (not assumed ones), and has been exercised end-to-end against two independent free emulators strict enough to have caught a genuine bug before it could reach real hardware. The IBM fallback, already implemented in Pass 4, is now verified one level deeper — actual device construction and actual circuit execution against a real IBM-provided fake backend, not just import checks.

What has *not* changed, and should not be overstated: neither backend has touched live hardware yet. Every claim in this document about Aquila or IBM says exactly that, plainly, next to the claim it's attached to. The honest next step is small and specific: `python scripts/hardware_validation.py --backend aquila --n_steps 5`, for real, before anything larger — and the pipeline is now actually ready for that to be the *first* thing that happens on real hardware, not a leap of faith.
