"""Measure demo fault-clearing voltage transients under switching variants."""

from __future__ import annotations

from dataclasses import replace
from math import sqrt

import numpy as np

from config import CONFIG, SimulationConfig
from fault_injection import scenario_for_fault_type
from simulate import simulate_scenario


FAULT_TYPES = ("LG", "LL", "LLG", "LLL")


def main() -> None:
    print(f"nominal steady-state phase peak: {sqrt(2.0) * CONFIG.source_ll_rms_v / sqrt(3.0):.1f} V")
    print(f"default integrator: {CONFIG.integrator}, default transition: {CONFIG.fault_transition_s * 1000.0:.1f} ms")
    print("transition sensitivity at 10 kHz:")
    for transition_s in (0.0, 0.0002, 0.0005, 0.001, 0.005, CONFIG.fault_transition_s, 0.050):
        config = replace(CONFIG, fault_transition_s=transition_s)
        _print_scenario_rows(config, f"transition={transition_s * 1000.0:.1f} ms")

    print("time-step sensitivity with instantaneous clearing:")
    for sampling_rate_hz in (5_000.0, 10_000.0, 20_000.0, 50_000.0):
        config = replace(CONFIG, sampling_rate_hz=sampling_rate_hz, fault_transition_s=0.0)
        _print_scenario_rows(config, f"fs={sampling_rate_hz:g} Hz, transition=0")

    print("time-step sensitivity with configured smooth clearing:")
    for sampling_rate_hz in (5_000.0, 10_000.0, 20_000.0, 50_000.0):
        config = replace(CONFIG, sampling_rate_hz=sampling_rate_hz)
        _print_scenario_rows(config, f"fs={sampling_rate_hz:g} Hz, transition={CONFIG.fault_transition_s * 1000.0:.1f} ms")

    print(
        "interpretation: instantaneous clearing is numerically fragile because "
        "the peak grows with sample rate; the configured smooth transition is "
        "time-step stable, so the remaining peak is the model's damped clearing "
        "transient rather than the original ideal-switch artifact."
    )


def _print_scenario_rows(config: SimulationConfig, label: str) -> None:
    values = []
    for index, fault_type in enumerate(FAULT_TYPES, start=1):
        scenario = scenario_for_fault_type(config, fault_type, index)
        result = simulate_scenario(scenario, config)
        metrics = _voltage_metrics(result)
        values.append(
            f"{fault_type}: clear_peak={metrics['clear_peak_v']:.1f} V "
            f"({metrics['clear_peak_pu']:.2f} pu), post_peak={metrics['post_peak_v']:.1f} V"
        )
    print(f"  {label}: " + "; ".join(values))


def _voltage_metrics(result: dict[str, object]) -> dict[str, float]:
    metadata = result["metadata"]
    time = np.asarray(result["time"], dtype=float)
    voltage = np.column_stack([result["Va"], result["Vb"], result["Vc"]]).astype(float)
    steady_peak = sqrt(2.0) * float(metadata["source_ll_rms_v"]) / sqrt(3.0)
    clear_t = float(metadata["t_end"])
    clear_mask = (time >= clear_t - 0.002) & (time <= clear_t + 0.010)
    post_mask = (time >= clear_t + 0.010) & (time <= min(float(metadata["t_end"]) + 0.050, time[-1]))
    if not np.any(post_mask):
        post_mask = time >= clear_t
    clear_peak = float(np.max(np.abs(voltage[clear_mask])))
    post_peak = float(np.max(np.abs(voltage[post_mask]))) if np.any(post_mask) else float("nan")
    return {
        "clear_peak_v": clear_peak,
        "clear_peak_pu": clear_peak / steady_peak,
        "post_peak_v": post_peak,
    }


if __name__ == "__main__":
    main()
