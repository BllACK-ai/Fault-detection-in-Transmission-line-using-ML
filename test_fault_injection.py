"""Focused regression checks for fault conductance construction."""

from __future__ import annotations

import numpy as np

from config import CONFIG
from fault_injection import FaultScenario, fault_conductance_matrix


def _scenario(fault_type: str, faulted_phases: str) -> FaultScenario:
    return FaultScenario(
        fault_type=fault_type,
        faulted_phases=faulted_phases,
        fault_section=1,
        fault_resistance_ohm=CONFIG.fault_resistance_min_ohm,
        fault_inception_angle_deg=0.0,
        fault_duration_s=CONFIG.default_fault_duration_s,
        t_start=CONFIG.fault_start_base_s,
        t_end=CONFIG.fault_start_base_s + CONFIG.default_fault_duration_s,
        load_p_total_w=CONFIG.load_p_total_w,
        load_power_factor=CONFIG.load_power_factor,
    )


def test_ll_fault_matrix_is_symmetric_phase_to_phase() -> None:
    matrix = fault_conductance_matrix(_scenario("LL", "AB"), CONFIG)
    g = 1.0 / CONFIG.fault_resistance_min_ohm
    expected = np.array(
        [
            [g, -g, 0.0],
            [-g, g, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(matrix, expected)
    np.testing.assert_allclose(matrix, matrix.T)


def test_llg_fault_matrix_ties_faulted_phases_and_ground() -> None:
    matrix = fault_conductance_matrix(_scenario("LLG", "AB"), CONFIG)
    g = 1.0 / CONFIG.fault_resistance_min_ohm
    expected = np.array(
        [
            [2.0 * g, -g, 0.0],
            [-g, 2.0 * g, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(matrix, expected)
    np.testing.assert_allclose(matrix, matrix.T)


if __name__ == "__main__":
    test_ll_fault_matrix_is_symmetric_phase_to_phase()
    test_llg_fault_matrix_ties_faulted_phases_and_ground()
    print("fault conductance checks: PASS")
