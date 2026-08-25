# Methods: Quantum Reservoir Computing for Multivariate Weather Forecasting

## Study design

We evaluate a transverse-field Ising quantum reservoir computer (QRC) for multivariate weather forecasting at Addis Ababa Bole airport. The design is strictly temporal: transformations are fit from the training chronology only, validation data are used for selection, and test data are held out until final evaluation. Experiments are performed separately for 6-hour and 24-hour horizons. Classical models receive the same projected inputs as the QRC, making the comparison a test of the reservoir and readout rather than of unequal feature processing.

The workflow separates three distinct activities: simulator calibration, primary forecasting, and noise/hardware robustness experiments. This distinction is important for reporting. A bounded calibration sweep identifies candidate settings without accessing the test set; the primary benchmark uses the selected configuration on complete chronological splits; a density-matrix noise screen or a hardware run is reported as a separate condition rather than silently treated as the primary result.

## Data source and temporal preprocessing

Hourly weather observations are obtained from the NOAA Integrated Surface Database for Addis Ababa Bole, Ethiopia (station `63450099999`) over 2018--2024. The predicted variables are temperature, relative humidity, sea-level pressure, and wind speed. Source records are parsed, quality-cleaned, and resampled onto a regular 6-hour grid. Short missing intervals are interpolated only up to a configured maximum gap; remaining unusable rows are excluded before forming learning examples. The logged full-data run contained 10,227 six-hourly time points.

The chronology is split without shuffling into 70% training, 15% validation, and 15% test partitions. Per-variable normalization statistics are estimated from the training partition alone and applied unchanged to validation and test data. Each example is a flattened window of 20 six-hourly observations, corresponding to five days of history. The label is the four-target observation one step (6 hours) or four steps (24 hours) ahead. Crucially, sliding windows are constructed independently within each temporal partition, so no window crosses a train/validation or validation/test boundary.

## Shared input projection

Each unprojected input window has 80 values (20 lags for each of four variables). A principal-component analysis (PCA) transformation is fit only on training windows and then applied to validation and test windows. In the primary setting, the number of retained components equals the number of QRC qubits (nine). The same fitted projector is supplied to the QRC, echo-state network (ESN), LSTM, and GRU. The PCA object is saved with the horizon-specific results, permitting reproduction of the exact input representation and preventing data leakage from held-out partitions.

## Quantum reservoir computer

The QRC is a fixed, untrained transverse-field Ising system,

$$
H = -J \sum_{\langle i,j \rangle} Z_i Z_j - h \sum_i X_i,
$$

where $J$ controls Ising couplings and $h$ controls the transverse field. A nearest-neighbour chain is the primary topology; a simulator-only all-to-all topology is assessed in a controlled topology comparison. Evolution is approximated using a first-order Trotter decomposition with four steps and total dimensionless evolution time one per input sample.

Projected inputs are converted to bounded rotation angles and injected through $R_Y$ rotations. The primary encoding is data reuploading: the same current input is re-encoded before each Trotter step. A standard single-injection mode is retained for a dedicated encoding ablation. The reservoir can also feed the preceding $\langle Z\rangle$ vector back into the next encoding input, providing a controlled recurrent memory mechanism. At each time point, the readout receives

$$
[\langle Z_1\rangle,\ldots,\langle Z_n\rangle,
\langle X_1\rangle,\ldots,\langle X_n\rangle],
$$

yielding $2n$ features for $n$ qubits. Devices and QNodes are built once per reservoir configuration and reused for all time steps. A 20-sample initial washout, or 10% for shorter calibration sequences, is discarded before readout fitting and scoring.

## QRC readout and warm-start conditions

The QRC uses ridge regression as a linear multi-output readout. Candidate models span a specified grid of ridge penalties and joint, independent, and ensemble multi-output strategies. The ensemble averages the best joint and independent candidate predictions. The retained readout is selected by mean validation RMSE, independently for each forecast horizon. The independent formulation fits separate target columns, but a candidate uses one ridge penalty shared across its targets; the pipeline does not claim independently tuned penalties per target.

Cold-start and warm-start readout conditions are both evaluated. Cold start solves ridge regression directly from QRC features. For warm start, hidden-state features from a fitted classical model--ESN by default, or optionally LSTM/GRU--are transformed through an SVD-based procedure to provide an initialization compatible with the QRC feature dimension. Ridge regression is then fit as a residual correction around that initialization. Both conditions reuse the same QRC state matrices. Therefore, any difference is attributable to the readout initialization rather than separate stochastic or numerical reservoir trajectories.

## Classical benchmark models

The baseline suite comprises persistence, ARIMA, ESN, LSTM, and GRU. Persistence uses the most recent target values in the input window. ARIMA is fitted independently for each target, using `pmdarima` stepwise order search where available and a bounded statsmodels AIC search otherwise; validation and test forecasts are generated with walk-forward updates. The ESN is a fixed random recurrent reservoir with a ridge readout and a 50-sample washout.

LSTM and GRU models are trained as streaming temporal models rather than as independent one-window examples. Their hidden state persists through the ordered training sequence, an initial 50-sample washout is excluded from loss and scoring, and truncated back-propagation through time is performed in contiguous chunks. This gives recurrent baselines genuine cross-sample memory comparable in purpose to the QRC and ESN. Missing optional dependencies and runtime failures are recorded in per-horizon baseline-status outputs instead of silently removing a baseline from a result table.

## Calibration and robustness experiments

Hamiltonian, topology, qubit-count, shot-budget, and noise studies use validation data only. Every candidate in a given sweep receives the same fixed contiguous chronological prefix from the training and validation partitions. Contiguous sampling is required because random row subsampling would change the reservoir dynamics. The standard sweep budget is no more than 800 training and 200 validation windows. Calibration sample counts are stored in CSV and JSON outputs alongside results.

For each forecast horizon, standard and data-reuploading encodings are compared over the same Cartesian product of $J$ and $h$ values, using identical calibration windows, qubit count, chain topology, washout, and ridge-penalty grid. The winning encoding and its corresponding $J,h$ setting are selected by validation RMSE and used only for that horizon's downstream QRC forecast. This paired design makes encoding comparison fair and allows the selected Hamiltonian to depend on encoding. Qubit scaling assesses configured reservoir sizes using the selected primary-horizon architecture, while shot ablation compares exact expectation values with finite-shot estimates. These are calibration/robustness measurements, not substitutes for the final held-out test evaluation.

Nonzero depolarizing noise requires density-matrix simulation, which is substantially more expensive than state-vector simulation. The noise experiment consequently uses a fixed 50-window training and 50-window validation calibration subset and is described as a simulator noise-robustness screen. It returns a screening value $p^*$, but the primary QRC benchmark, qubit-scaling experiment, and shot ablation use $p=0$ by default. Applying the calibrated noise value end-to-end requires an explicit command-line opt-in and must be reported as a separate noisy-QRC protocol.

## Evaluation, reproducibility, and hardware scope

We report RMSE and MAE per target and their mean across targets. Valid prediction time is also computed on the 6-hour grid using the configured error threshold and Lyapunov-time reference. Metrics are first computed in the normalized modeling space; where scaler metadata are available, results are also expressed in physical units. Model-specific washout offsets are respected when matching predictions and labels.

For reproducibility, the pipeline saves model-result objects, selected QRC readouts, configuration files, PCA projectors, readout-selection logs, sweep tables, and baseline-status reports. Paper tables should identify the data period, split counts, horizon, random seed, selected configuration, and whether the result comes from the noiseless simulator, a noise screen, an emulator, or a live device.

The simulator is the primary evaluation environment. IBM hardware support executes the gate-model circuit through PennyLane/Qiskit Runtime. QuEra Aquila support is an independent analog Rydberg implementation: $J$ is mapped to atom spacing, $h$ to global Rabi amplitude, and inputs to local detuning. Aquila requires separate Z-like and rotated X-like measurement programs. Geometry and shot limits are checked, and the realized coupling is logged if a requested geometry must be clamped. Aquila all-to-all coupling is a compact-ring approximation, not literal uniform all-to-all coupling. The physical scaling constants are documented design parameters rather than empirically calibrated values. No hardware result should be represented as live-QPU validated until a small, logged real-device experiment has been completed and compared with its simulator or emulator control.
