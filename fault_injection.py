"""Fault scenario sampling and fault conductance construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import pi
from typing import Iterable

import numpy as np

from config import SimulationConfig


FAULT_TYPES = ("Normal", "LG", "LL", "LLG", "LLL")
PHASE_INDEX = {"A": 0, "B": 1, "C": 2}
PHASE_PAIRS = ("AB", "BC", "CA")


@dataclass(frozen=True)
class FaultScenario:
    """Labels and timing for one simulation run."""

    fault_type: str
    faulted_phases: str
    fault_section: int | None
    fault_resistance_ohm: float | None
    fault_inception_angle_deg: float | None
    fault_duration_s: float
    t_start: float
    t_end: float
    load_p_total_w: float
    load_power_factor: float

    def metadata(self) -> dict[str, object]:
        return asdict(self)


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_fault_scenario(config: SimulationConfig, rng: np.random.Generator) -> FaultScenario:
    """Randomly sample a labeled scenario for dataset generation."""

    fault_type = str(rng.choice(FAULT_TYPES))
    p_scale = rng.uniform(
        1.0 - config.load_p_variation_fraction,
        1.0 + config.load_p_variation_fraction,
    )
    pf = float(
        np.clip(
            config.load_power_factor + rng.uniform(-config.load_pf_variation, config.load_pf_variation),
            0.75,
            0.99,
        )
    )
    angle_deg = float(rng.uniform(0.0, 180.0))
    inception_delay_s = angle_deg / 360.0 / config.frequency_hz
    t_start = config.fault_start_base_s + inception_delay_s
    duration = float(rng.uniform(config.fault_duration_min_s, config.fault_duration_max_s))
    t_end = min(config.t_end_s, t_start + duration)

    if fault_type == "Normal":
        return FaultScenario(
            fault_type="Normal",
            faulted_phases="",
            fault_section=None,
            fault_resistance_ohm=None,
            fault_inception_angle_deg=None,
            fault_duration_s=0.0,
            t_start=config.t_end_s + 1.0,
            t_end=config.t_end_s + 1.0,
            load_p_total_w=config.load_p_total_w * p_scale,
            load_power_factor=pf,
        )

    if fault_type == "LG":
        faulted = str(rng.choice(("A", "B", "C")))
    elif fault_type in {"LL", "LLG"}:
        faulted = str(rng.choice(PHASE_PAIRS))
    else:
        faulted = "ABC"

    return FaultScenario(
        fault_type=fault_type,
        faulted_phases=faulted,
        fault_section=int(rng.integers(1, config.n_sections + 1)),
        fault_resistance_ohm=_log_uniform(
            rng,
            config.fault_resistance_min_ohm,
            config.fault_resistance_max_ohm,
        ),
        fault_inception_angle_deg=angle_deg,
        fault_duration_s=duration,
        t_start=t_start,
        t_end=t_end,
        load_p_total_w=config.load_p_total_w * p_scale,
        load_power_factor=pf,
    )


def scenario_for_fault_type(config: SimulationConfig, fault_type: str, index: int = 0) -> FaultScenario:
    """Deterministic scenario used by demo and validation code."""

    if fault_type not in FAULT_TYPES:
        raise ValueError(f"unsupported fault_type {fault_type!r}")
    phases_by_type = {
        "Normal": "",
        "LG": "A",
        "LL": "AB",
        "LLG": "AB",
        "LLL": "ABC",
    }
    section = None if fault_type == "Normal" else 1
    angle_deg = 45.0 + 20.0 * index
    t_start = config.fault_start_base_s + angle_deg / 360.0 / config.frequency_hz
    duration = config.default_fault_duration_s
    if fault_type == "Normal":
        t_start = config.t_end_s + 1.0
    return FaultScenario(
        fault_type=fault_type,
        faulted_phases=phases_by_type[fault_type],
        fault_section=section,
        fault_resistance_ohm=None if fault_type == "Normal" else config.fault_resistance_min_ohm,
        fault_inception_angle_deg=None if fault_type == "Normal" else angle_deg,
        fault_duration_s=0.0 if fault_type == "Normal" else duration,
        t_start=t_start,
        t_end=t_start + duration,
        load_p_total_w=config.load_p_total_w,
        load_power_factor=config.load_power_factor,
    )


def fault_is_active(t_s: float, scenario: FaultScenario) -> bool:
    return scenario.fault_type != "Normal" and scenario.t_start <= t_s < scenario.t_end


def phase_indices(phase_string: str) -> list[int]:
    return [PHASE_INDEX[phase] for phase in phase_string]


def fault_conductance_matrix(
    scenario: FaultScenario,
    config: SimulationConfig,
) -> np.ndarray | None:
    """Build the ABC shunt conductance matrix for the faulted node.

    LG faults add phase-to-ground conductance. LL faults add a phase-to-phase
    conductance, which injects equal and opposite currents based on voltage
    difference. LLG faults combine that phase-to-phase tie with equal shunts
    from the two faulted phases to ground. LLL faults connect all phase pairs
    and, by default, each phase to ground through the same finite resistance to
    represent a three-phase-to-ground fault.
    """

    if scenario.fault_type == "Normal":
        return None
    rf = max(float(scenario.fault_resistance_ohm or config.fault_resistance_min_ohm), 1e-6)
    g = 1.0 / rf
    matrix = np.zeros((3, 3), dtype=float)

    if scenario.fault_type == "LG":
        matrix[PHASE_INDEX[scenario.faulted_phases], PHASE_INDEX[scenario.faulted_phases]] += g
    elif scenario.fault_type == "LL":
        _add_phase_to_phase_conductance(matrix, phase_indices(scenario.faulted_phases), g)
    elif scenario.fault_type == "LLG":
        phases = phase_indices(scenario.faulted_phases)
        _add_phase_to_phase_conductance(matrix, phases, g)
        for phase in phases:
            matrix[phase, phase] += g
    elif scenario.fault_type == "LLL":
        for pair in ((0, 1), (1, 2), (2, 0)):
            _add_phase_to_phase_conductance(matrix, pair, g)
        if config.lll_fault_to_ground:
            matrix += np.eye(3) * g
    else:
        raise ValueError(f"unsupported fault type {scenario.fault_type!r}")
    return matrix


def _add_phase_to_phase_conductance(matrix: np.ndarray, phases: Iterable[int], conductance_s: float) -> None:
    a, b = list(phases)
    matrix[a, a] += conductance_s
    matrix[b, b] += conductance_s
    matrix[a, b] -= conductance_s
    matrix[b, a] -= conductance_s
