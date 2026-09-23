# Transmission Line Fault Detection and Location using Machine Learning

A physics-based simulation and machine learning pipeline for detecting, classifying, and locating faults on a three-phase transmission line, built end-to-end: circuit-level ODE simulation → randomized fault injection → feature engineering → Random Forest classification → noise-robustness evaluation.

## Overview

Power system relays need to answer two questions fast: **what kind of fault happened**, and **where on the line did it happen**. This project builds both answers from simulated relay measurements (three-phase voltage and current at the sending end) rather than lab hardware, so large, precisely-labeled datasets can be generated cheaply.

The pipeline:

1. **Circuit simulation** (`network_model.py`, `simulate.py`) — a 6-section three-phase radial R-L line model, with mutual phase coupling, a weak Thevenin source, and a balanced RL load, solved as a stiff ODE system.
2. **Fault injection** (`fault_injection.py`) — randomly samples fault type (LG, LL, LLG, LLL, or Normal), location, fault resistance, inception angle, and duration for each simulation run.
3. **Dataset generation** (`dataset_generator.py`, `build_dataset.py`) — 3,000 simulated runs, balanced across 5 classes (600 each: Normal, LG, LL, LLG, LLL).
4. **Feature extraction** (`feature_extraction.py`) — time-domain statistics (RMS, mean, std, skewness, kurtosis, peak-to-peak) per channel, plus symmetrical components (V0/V1/V2, I0/I1/I2) and per-phase impedance magnitude/angle.
5. **Model training** (`train_fault_classifier.py`, `train_fault_location_classifier.py`) — two Random Forest classifiers (200 trees each): one for fault type, one for fault location (which of 6 line sections).
6. **Robustness evaluation** (`evaluate_noise_robustness.py`) — re-tests both models after injecting measurement noise at controlled SNR levels (40/30/20 dB), simulating real sensor noise rather than reporting clean-signal accuracy only.

## Results

**Fault-type classification**

| Condition | Test accuracy |
|---|---|
| Clean signal | ~100% |
| 40 dB SNR | ~100% |
| 30 dB SNR | 97.1% |
| 20 dB SNR | 72.9% |

**Fault-location classification** (6-way, chance ≈ 16.7%)

| Condition | Test accuracy |
|---|---|
| Clean signal | 62.8% |
| 40 dB SNR | 61.8% |
| 30 dB SNR | 59.9% |
| 20 dB SNR | 58.6% |

Location is the harder problem — accuracy well above chance but far from perfect, and it degrades more gracefully under noise than fault-type accuracy does, since it relies more on impedance-angle features than raw waveform shape.

**Most important features** — fault-type classification leans on zero/negative-sequence current and voltage magnitudes (`I0_rms`, `V0_rms`, `I2_rms`, `V0/V1` ratio); fault-location classification leans on per-phase impedance angle (`Za_angle_deg`, `Zb_angle_deg`, `Zc_angle_deg`).

The training script also runs a built-in sanity check that flags a model if its accuracy looks suspiciously close to 100% across all classes — a guard against silent data leakage rather than a reported metric to chase.

## Repository structure

```
config.py                              # simulation & dataset parameters
network_model.py                       # ODE state-space model of the line
fault_injection.py                     # randomized fault scenario sampling
simulate.py                            # runs one simulation scenario
dataset_generator.py / build_dataset.py# dataset generation
feature_extraction.py                  # waveform -> feature vector
train_fault_classifier.py              # fault-type Random Forest
train_fault_location_classifier.py     # fault-location Random Forest
evaluate_noise_robustness.py           # SNR sweep evaluation
analyze_clearing_transients.py         # post-hoc transient analysis
fault_classifier_rf.joblib             # trained fault-type model
fault_location_rf.joblib               # trained fault-location model
fault_dataset.csv                      # extracted feature dataset (3,000 rows)
```

## Running it

```bash
python dataset_generator.py --runs 3000        # generate simulated runs
python build_dataset.py                        # extract features to fault_dataset.csv
python train_fault_classifier.py                # train + evaluate fault-type model
python train_fault_location_classifier.py        # train + evaluate fault-location model
python evaluate_noise_robustness.py              # SNR robustness sweep
```

## Notable design choices

- **Backward Euler integration** was chosen over higher-order methods because the low shunt capacitance and low-impedance faults create a numerically stiff system; backward Euler's damping suppresses non-physical switching spikes without the cost of adaptive stiff solvers.
- **Log-uniform sampling of fault resistance** (0.3–100 Ω) avoids over-representing near-zero-resistance "easy" faults, which would otherwise dominate a uniform sample and inflate reported accuracy.
- **Noise is injected into raw waveforms before feature extraction**, not into the final feature vector, so the robustness numbers reflect realistic sensor noise rather than an easier, feature-space approximation.

## Limitations

- Single radial line topology (6 sections, one source, one load) — not a meshed network.
- Location accuracy (62.8%) reflects a genuinely hard 6-way discrimination problem on this topology; it is reported honestly rather than tuned for a headline number.
- All data is simulated; no field or lab hardware data is used.
