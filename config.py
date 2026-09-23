"""Configuration for the three-phase fault simulation.

All physical and dataset-generation parameters are collected here so the
simulation code does not hide engineering constants inside the numerical logic.
Values are intentionally conservative for a low-spec machine and can be edited
before generating a large dataset.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    """Physical and numerical parameters for one simulation campaign."""

    # Source
    source_ll_rms_v: float = 11_000.0
    frequency_hz: float = 50.0
    # Per-phase Thevenin impedance between the ideal internal EMF and the
    # sending-end relay bus. Defaults are roughly an 11 kV, 75 MVA short-
    # circuit source with X/R near 10: |Z| = Vll^2 / Ssc = 1.61 ohm. This weak
    # but plausible grid source makes relay-bus voltage sag visible during
    # downstream faults.
    source_r_ohm: float = 0.16
    source_x_ohm: float = 1.60

    # Radial line model: N identical coupled series R-L sections.
    n_sections: int = 6
    section_length_km: float = 2.0
    line_r_self_ohm_per_km: float = 0.22
    line_r_mutual_ohm_per_km: float = 0.035
    line_l_self_h_per_km: float = 1.25e-3
    line_l_mutual_h_per_km: float = 0.42e-3

    # Small phase-to-ground node capacitance makes the nodal ladder an ODE
    # rather than a DAE. It approximates distributed line shunt capacitance;
    # keep the floor near the physical section capacitance so the healthy
    # current is not dominated by artificial charging current.
    line_c_shunt_f_per_km: float = 12e-9
    min_node_capacitance_f: float = 25e-9

    # Balanced three-phase RL load.
    load_p_total_w: float = 1_000_000.0
    load_power_factor: float = 0.90
    load_p_variation_fraction: float = 0.10
    load_pf_variation: float = 0.04

    # Sampling and simulation window.
    sampling_rate_hz: float = 10_000.0
    t_end_s: float = 0.5
    fault_start_base_s: float = 0.20
    default_fault_duration_s: float = 0.12
    fault_duration_min_s: float = 0.05
    fault_duration_max_s: float = 0.30
    fault_transition_s: float = 0.020

    # Fault randomization. A finite minimum avoids singular matrices and
    # pathological switching stiffness while still producing near-bolted,
    # low-impedance fault currents.
    fault_resistance_min_ohm: float = 0.30
    fault_resistance_max_ohm: float = 100.0
    lll_fault_to_ground: bool = True

    # Integrator choices: "backward_euler", "trapezoidal", "rk4",
    # "solve_ivp_radau", or "solve_ivp_bdf". Backward Euler is the default
    # because the low physical shunt capacitance and low fault resistance create
    # stiff high-frequency modes; its numerical damping suppresses nonphysical
    # switching spikes while remaining lightweight for dataset generation.
    integrator: str = "backward_euler"
    rtol: float = 1e-5
    atol: float = 1e-7

    # Dataset/demo output.
    output_dir: str = "simulation_output"
    dataset_dir: str = "dataset"
    demo_dir: str = "demo"
    metadata_filename: str = "metadata.csv"
    random_seed: int = 12345


CONFIG = SimulationConfig()
