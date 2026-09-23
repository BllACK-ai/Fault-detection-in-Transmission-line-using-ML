"""Feature extraction for three-phase fault diagnosis runs."""

from __future__ import annotations

from math import sqrt
from typing import Mapping

import numpy as np


CHANNEL_NAMES = ("Va", "Vb", "Vc", "Ia", "Ib", "Ic")
TIME_STAT_NAMES = ("rms", "mean", "std", "skewness", "kurtosis", "peak_to_peak")
FEATURE_COLUMNS = tuple(
    f"{channel}_{stat}"
    for channel in CHANNEL_NAMES
    for stat in TIME_STAT_NAMES
) + (
    "V0_rms",
    "V1_rms",
    "V2_rms",
    "I0_rms",
    "I1_rms",
    "I2_rms",
    "V2_over_V1",
    "V0_over_V1",
    "Za_mag",
    "Zb_mag",
    "Zc_mag",
    "Za_angle_deg",
    "Zb_angle_deg",
    "Zc_angle_deg",
    "Z1_mag",
)
LABEL_COLUMNS = (
    "fault_type",
    "faulted_phases",
    "fault_section",
    "fault_resistance_ohm",
    "fault_inception_angle_deg",
    "fault_duration_s",
)
OUTPUT_COLUMNS = FEATURE_COLUMNS + LABEL_COLUMNS


def select_feature_window(
    t: np.ndarray,
    metadata: Mapping[str, object],
    fs: float,
    normal_window_duration_s: float = 0.12,
) -> np.ndarray:
    """Return the sample mask used for feature extraction.

    Faulted runs use exactly the fault-on interval because that is where the
    diagnostic signature exists. Normal runs use an equal-duration window in
    the middle of the record so their features are computed from a comparable
    number of healthy samples.
    """

    time = np.asarray(t, dtype=float)
    if time.ndim != 1 or len(time) == 0:
        raise ValueError("t must be a non-empty one-dimensional array")
    fault_type = str(metadata.get("fault_type", "Normal"))
    if fault_type != "Normal":
        start = float(metadata["t_start"])
        end = float(metadata["t_end"])
    else:
        duration = float(metadata.get("typical_fault_duration_s", normal_window_duration_s))
        duration = max(duration, 2.0 / float(fs))
        center = 0.5 * (float(time[0]) + float(time[-1]))
        start = center - 0.5 * duration
        end = center + 0.5 * duration

    mask = (time >= start) & (time < end)
    if not np.any(mask):
        raise ValueError(f"empty feature window for {fault_type}: start={start}, end={end}")
    return mask


def extract_features(
    t,
    Va,
    Vb,
    Vc,
    Ia,
    Ib,
    Ic,
    metadata: Mapping[str, object],
    fs: float,
) -> dict[str, object]:
    """Extract one flat feature row from relay-bus voltage/current waveforms.

    Sequence components are computed from fundamental-frequency phasors, not
    instantaneous samples. This matches protection practice: positive,
    negative, and zero sequence quantities are defined for sinusoidal phasors,
    while direct instantaneous transformation would mix the fundamental with
    switching transients, offsets, and higher-frequency numerical content.
    """

    time = np.asarray(t, dtype=float)
    channels = {
        "Va": np.asarray(Va, dtype=float),
        "Vb": np.asarray(Vb, dtype=float),
        "Vc": np.asarray(Vc, dtype=float),
        "Ia": np.asarray(Ia, dtype=float),
        "Ib": np.asarray(Ib, dtype=float),
        "Ic": np.asarray(Ic, dtype=float),
    }
    if any(values.shape != time.shape for values in channels.values()):
        raise ValueError("all waveform arrays must have the same shape as t")

    mask = select_feature_window(time, metadata, fs)
    row: dict[str, object] = {}
    for name, values in channels.items():
        row.update(_time_statistics(name, values[mask]))

    frequency_hz = float(metadata.get("frequency_hz", 50.0))
    v_phasors = np.array(
        [
            _fundamental_phasor(time[mask], channels["Va"][mask], frequency_hz),
            _fundamental_phasor(time[mask], channels["Vb"][mask], frequency_hz),
            _fundamental_phasor(time[mask], channels["Vc"][mask], frequency_hz),
        ],
        dtype=complex,
    )
    i_phasors = np.array(
        [
            _fundamental_phasor(time[mask], channels["Ia"][mask], frequency_hz),
            _fundamental_phasor(time[mask], channels["Ib"][mask], frequency_hz),
            _fundamental_phasor(time[mask], channels["Ic"][mask], frequency_hz),
        ],
        dtype=complex,
    )
    v0, v1, v2 = _symmetrical_components(v_phasors)
    i0, i1, i2 = _symmetrical_components(i_phasors)
    row["V0_rms"] = float(abs(v0))
    row["V1_rms"] = float(abs(v1))
    row["V2_rms"] = float(abs(v2))
    row["I0_rms"] = float(abs(i0))
    row["I1_rms"] = float(abs(i1))
    row["I2_rms"] = float(abs(i2))
    row["V2_over_V1"] = _safe_ratio(abs(v2), abs(v1))
    row["V0_over_V1"] = _safe_ratio(abs(v0), abs(v1))
    za, zb, zc = [_safe_impedance(v_phasor, i_phasor) for v_phasor, i_phasor in zip(v_phasors, i_phasors)]
    z1 = _safe_impedance(v1, i1)
    row["Za_mag"] = float(abs(za))
    row["Zb_mag"] = float(abs(zb))
    row["Zc_mag"] = float(abs(zc))
    row["Za_angle_deg"] = float(np.degrees(np.angle(za)))
    row["Zb_angle_deg"] = float(np.degrees(np.angle(zb)))
    row["Zc_angle_deg"] = float(np.degrees(np.angle(zc)))
    row["Z1_mag"] = float(abs(z1))

    row.update(_label_values(metadata))
    return row


def feature_columns() -> tuple[str, ...]:
    return FEATURE_COLUMNS


def output_columns() -> tuple[str, ...]:
    return OUTPUT_COLUMNS


def _time_statistics(prefix: str, values: np.ndarray) -> dict[str, float]:
    mean = float(np.mean(values))
    centered = values - mean
    std = float(np.std(values))
    if std <= 1e-12:
        skewness = 0.0
        kurtosis = 0.0
    else:
        normalized = centered / std
        skewness = float(np.mean(normalized**3))
        kurtosis = float(np.mean(normalized**4) - 3.0)
    return {
        f"{prefix}_rms": float(np.sqrt(np.mean(values**2))),
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_skewness": skewness,
        f"{prefix}_kurtosis": kurtosis,
        f"{prefix}_peak_to_peak": float(np.max(values) - np.min(values)),
    }


def _fundamental_phasor(time: np.ndarray, values: np.ndarray, frequency_hz: float) -> complex:
    centered_time = time - float(time[0])
    kernel = np.exp(-2j * np.pi * frequency_hz * centered_time)
    peak_phasor = 2.0 * np.mean(values * kernel)
    return complex(peak_phasor / sqrt(2.0))


def _symmetrical_components(phase_phasors: np.ndarray) -> tuple[complex, complex, complex]:
    a = np.exp(2j * np.pi / 3.0)
    va, vb, vc = phase_phasors
    zero = (va + vb + vc) / 3.0
    positive = (va + a * vb + a**2 * vc) / 3.0
    negative = (va + a**2 * vb + a * vc) / 3.0
    return zero, positive, negative


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 1e-12:
        return 0.0
    return float(numerator / denominator)


def _safe_impedance(voltage: complex, current: complex) -> complex:
    if abs(current) <= 1e-12:
        return 0.0 + 0.0j
    return complex(voltage / current)


def _label_values(metadata: Mapping[str, object]) -> dict[str, object]:
    fault_type = str(metadata.get("fault_type", "Normal"))
    faulted_phases = metadata.get("faulted_phases") or "None"
    fault_section = metadata.get("fault_section")
    # These fields are retained for audit/stratification only. They should not
    # be included in the classifier feature matrix, because they would leak
    # information unavailable at inference time.
    return {
        "fault_type": fault_type,
        "faulted_phases": faulted_phases,
        "fault_section": fault_section if fault_section is not None else "None",
        "fault_resistance_ohm": _none_label(metadata.get("fault_resistance_ohm")),
        "fault_inception_angle_deg": _none_label(metadata.get("fault_inception_angle_deg")),
        "fault_duration_s": _none_label(metadata.get("fault_duration_s")),
    }


def _none_label(value: object) -> object:
    return "None" if value is None else value
