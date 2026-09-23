"""ABC-domain state-space model for a radial three-phase R-L line.

The network is represented as:

    ideal internal EMF -> source Thevenin R-L -> sending bus
        -> N coupled series R-L sections -> balanced RL load

Each section has a 3x3 resistance matrix Rabc and inductance matrix Labc:
diagonal entries are the phase self terms and off-diagonal entries are mutual
terms. This standard symmetrical phase matrix captures phase coupling during
unbalanced faults instead of treating A, B, and C as independent circuits.

States are ordered as:
    [source_current, line_currents for sections 1..N,
     node_voltages for sending bus and downstream nodes 1..N, load_currents]

Source current states obey Ls * dis/dt = e_internal - v_send - Rs*is.
Line current states obey Labc * di/dt = v_upstream - v_downstream - Rabc*i.
Node voltage states obey C * dv/dt = i_in - i_out - i_load - i_fault.
Load current states obey Lload * di_load/dt = v_terminal - Rload*i_load.

The small node capacitance represents lumped line shunt capacitance and turns
the nodal model into an ODE. Without it, the intermediate node voltages would be
algebraic variables, which is a DAE and less convenient for low-cost dataset
generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

import numpy as np

from config import SimulationConfig


PHASES = ("A", "B", "C")


@dataclass(frozen=True)
class StateLayout:
    """Index helpers for the flattened ODE state vector."""

    n_sections: int

    @property
    def n_source_current(self) -> int:
        return 3

    @property
    def n_line_current(self) -> int:
        return 3 * self.n_sections

    @property
    def n_node_voltage(self) -> int:
        return 3 * (self.n_sections + 1)

    @property
    def n_state(self) -> int:
        return self.n_source_current + self.n_line_current + self.n_node_voltage + 3

    @property
    def source_current_slice(self) -> slice:
        return slice(0, 3)

    def current_slice(self, section_zero_based: int) -> slice:
        start = self.n_source_current + 3 * section_zero_based
        return slice(start, start + 3)

    def voltage_slice(self, node_zero_based: int) -> slice:
        start = self.n_source_current + self.n_line_current + 3 * node_zero_based
        return slice(start, start + 3)

    @property
    def sending_voltage_slice(self) -> slice:
        return self.voltage_slice(0)

    @property
    def load_current_slice(self) -> slice:
        start = self.n_source_current + self.n_line_current + self.n_node_voltage
        return slice(start, start + 3)


@dataclass(frozen=True)
class NetworkModel:
    """Precomputed matrices and electrical constants for simulation."""

    config: SimulationConfig
    layout: StateLayout
    r_line_abc: np.ndarray
    l_line_abc: np.ndarray
    inv_l_line_abc: np.ndarray
    r_source_abc: np.ndarray
    l_source_abc: np.ndarray
    inv_l_source_abc: np.ndarray
    node_capacitance_f: float
    load_r_ohm: float
    load_l_h: float
    source_phase_peak_v: float
    omega_rad_s: float


def symmetrical_matrix(self_value: float, mutual_value: float) -> np.ndarray:
    """Return a balanced 3x3 phase-domain matrix with equal mutual terms."""

    matrix = np.full((3, 3), mutual_value, dtype=float)
    np.fill_diagonal(matrix, self_value)
    return matrix


def load_rl_from_power(config: SimulationConfig, p_total_w: float, power_factor: float) -> tuple[float, float]:
    """Convert balanced three-phase P and lagging PF to per-phase series RL.

    The load is represented as three identical phase-to-ground series R-L
    branches. At nominal voltage, each branch has impedance magnitude
    |Z| = V_phase_rms / I_phase_rms, with R = |Z|*pf and X = |Z|*sin(phi).
    """

    if not 0.05 < power_factor <= 1.0:
        raise ValueError("load power factor must be in (0.05, 1.0]")
    v_phase_rms = config.source_ll_rms_v / sqrt(3.0)
    i_phase_rms = p_total_w / (3.0 * v_phase_rms * power_factor)
    z_mag = v_phase_rms / i_phase_rms
    r_load = z_mag * power_factor
    x_load = z_mag * sqrt(max(0.0, 1.0 - power_factor * power_factor))
    l_load = x_load / (2.0 * pi * config.frequency_hz)
    return r_load, max(l_load, 1e-6)


def build_network_model(
    config: SimulationConfig,
    load_p_total_w: float | None = None,
    load_power_factor: float | None = None,
) -> NetworkModel:
    """Build reusable network constants from configuration values."""

    p_total = config.load_p_total_w if load_p_total_w is None else load_p_total_w
    pf = config.load_power_factor if load_power_factor is None else load_power_factor
    length = config.section_length_km

    r_line = symmetrical_matrix(
        config.line_r_self_ohm_per_km * length,
        config.line_r_mutual_ohm_per_km * length,
    )
    l_line = symmetrical_matrix(
        config.line_l_self_h_per_km * length,
        config.line_l_mutual_h_per_km * length,
    )
    source_l_h = config.source_x_ohm / (2.0 * pi * config.frequency_hz)
    r_source = symmetrical_matrix(config.source_r_ohm, 0.0)
    l_source = symmetrical_matrix(max(source_l_h, 1e-6), 0.0)
    r_load, l_load = load_rl_from_power(config, p_total, pf)
    c_node = max(config.line_c_shunt_f_per_km * length, config.min_node_capacitance_f)

    return NetworkModel(
        config=config,
        layout=StateLayout(config.n_sections),
        r_line_abc=r_line,
        l_line_abc=l_line,
        inv_l_line_abc=np.linalg.inv(l_line),
        r_source_abc=r_source,
        l_source_abc=l_source,
        inv_l_source_abc=np.linalg.inv(l_source),
        node_capacitance_f=c_node,
        load_r_ohm=r_load,
        load_l_h=l_load,
        source_phase_peak_v=sqrt(2.0) * config.source_ll_rms_v / sqrt(3.0),
        omega_rad_s=2.0 * pi * config.frequency_hz,
    )


def balanced_source_voltage(t_s: float, model: NetworkModel) -> np.ndarray:
    """Return ideal phase-to-ground source voltages [Va, Vb, Vc]."""

    angle = model.omega_rad_s * t_s
    return model.source_phase_peak_v * np.array(
        [
            np.sin(angle),
            np.sin(angle - 2.0 * np.pi / 3.0),
            np.sin(angle + 2.0 * np.pi / 3.0),
        ],
        dtype=float,
    )


def state_derivative(
    t_s: float,
    y: np.ndarray,
    model: NetworkModel,
    fault_conductance_abc: np.ndarray | None = None,
    fault_section: int | None = None,
) -> np.ndarray:
    """Compute dy/dt for the present topology.

    Fault switching is implemented as a finite shunt conductance matrix applied
    at the downstream node of the selected section. This is a topology change
    between pre-fault, during-fault, and post-fault models. A finite Rf keeps
    the ODE well-defined for low-impedance faults.
    """

    layout = model.layout
    dydt = np.zeros_like(y)
    internal_v = balanced_source_voltage(t_s, model)

    # Source Thevenin branch from the ideal internal EMF to the sending bus.
    source_i = y[layout.source_current_slice]
    sending_v = y[layout.sending_voltage_slice]
    dydt[layout.source_current_slice] = model.inv_l_source_abc @ (
        internal_v - sending_v - model.r_source_abc @ source_i
    )

    # Series line current equations.
    for section in range(model.config.n_sections):
        i_slice = layout.current_slice(section)
        i_line = y[i_slice]
        upstream_v = y[layout.voltage_slice(section)]
        downstream_v = y[layout.voltage_slice(section + 1)]
        dydt[i_slice] = model.inv_l_line_abc @ (upstream_v - downstream_v - model.r_line_abc @ i_line)

    # Node voltage equations.
    for node in range(model.config.n_sections + 1):
        v_slice = layout.voltage_slice(node)
        if node == 0:
            incoming_i = y[layout.source_current_slice]
        else:
            incoming_i = y[layout.current_slice(node - 1)]
        outgoing_i = (
            y[layout.current_slice(node)]
            if node < model.config.n_sections
            else np.zeros(3, dtype=float)
        )
        shunt_i = np.zeros(3, dtype=float)
        if node == model.config.n_sections:
            shunt_i += y[layout.load_current_slice]
        if (
            fault_conductance_abc is not None
            and fault_section is not None
            and node == fault_section
        ):
            shunt_i += fault_conductance_abc @ y[v_slice]
        dydt[v_slice] = (incoming_i - outgoing_i - shunt_i) / model.node_capacitance_f

    # Balanced series RL load at the final node.
    terminal_v = y[layout.voltage_slice(model.config.n_sections)]
    load_i = y[layout.load_current_slice]
    dydt[layout.load_current_slice] = (terminal_v - model.load_r_ohm * load_i) / model.load_l_h
    return dydt


def build_state_space_matrices(
    model: NetworkModel,
    fault_conductance_abc: np.ndarray | None = None,
    fault_section: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return A, B for dy/dt = A*y + B*v_source(t).

    Precomputing these matrices is much faster than assembling the same linear
    equations inside every RK4 sub-step. Faulted and healthy periods use
    different A matrices; the source-injection matrix B is unchanged.
    """

    layout = model.layout
    n_state = layout.n_state
    matrix_a = np.zeros((n_state, n_state), dtype=float)
    matrix_b = np.zeros((n_state, 3), dtype=float)
    identity = np.eye(3)

    source_slice = layout.source_current_slice
    sending_slice = layout.sending_voltage_slice
    matrix_a[source_slice, source_slice] += -model.inv_l_source_abc @ model.r_source_abc
    matrix_a[source_slice, sending_slice] += -model.inv_l_source_abc
    matrix_b[source_slice, :] += model.inv_l_source_abc

    for section in range(model.config.n_sections):
        i_slice = layout.current_slice(section)
        v_up_slice = layout.voltage_slice(section)
        v_down_slice = layout.voltage_slice(section + 1)
        matrix_a[i_slice, i_slice] += -model.inv_l_line_abc @ model.r_line_abc
        matrix_a[i_slice, v_up_slice] += model.inv_l_line_abc
        matrix_a[i_slice, v_down_slice] += -model.inv_l_line_abc

    for node in range(model.config.n_sections + 1):
        v_slice = layout.voltage_slice(node)
        if node == 0:
            matrix_a[v_slice, layout.source_current_slice] += identity / model.node_capacitance_f
        else:
            matrix_a[v_slice, layout.current_slice(node - 1)] += identity / model.node_capacitance_f
        if node < model.config.n_sections:
            matrix_a[v_slice, layout.current_slice(node)] += -identity / model.node_capacitance_f
        else:
            matrix_a[v_slice, layout.load_current_slice] += -identity / model.node_capacitance_f
        if (
            fault_conductance_abc is not None
            and fault_section is not None
            and node == fault_section
        ):
            matrix_a[v_slice, v_slice] += -fault_conductance_abc / model.node_capacitance_f

    load_slice = layout.load_current_slice
    terminal_v_slice = layout.voltage_slice(model.config.n_sections)
    matrix_a[load_slice, terminal_v_slice] += identity / model.load_l_h
    matrix_a[load_slice, load_slice] += -identity * (model.load_r_ohm / model.load_l_h)
    return matrix_a, matrix_b


def matrix_state_derivative(t_s: float, y: np.ndarray, model: NetworkModel, matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    """Fast derivative using precomputed state-space matrices."""

    return matrix_a @ y + matrix_b @ balanced_source_voltage(t_s, model)


def initial_state(config: SimulationConfig, model: NetworkModel) -> np.ndarray:
    """Return the healthy sinusoidal steady-state at t=0.

    The linear healthy model is y' = A*y + B*e(t). With a balanced sinusoidal
    internal EMF e(t) = s*sin(wt) + c*cos(wt), the steady state has the same
    form y(t) = p*sin(wt) + q*cos(wt). Solving for p and q gives q as the
    initial state at t=0. This avoids long startup transients in the relay data.
    """

    matrix_a, matrix_b = build_state_space_matrices(model)
    omega = model.omega_rad_s
    identity = np.eye(model.layout.n_state)
    phase_angles = np.array([0.0, -2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0])
    sine_coeff = model.source_phase_peak_v * np.cos(phase_angles)
    cosine_coeff = model.source_phase_peak_v * np.sin(phase_angles)

    block = np.block(
        [
            [matrix_a, omega * identity],
            [-omega * identity, matrix_a],
        ]
    )
    rhs = -np.concatenate([matrix_b @ sine_coeff, matrix_b @ cosine_coeff])
    solution = np.linalg.solve(block, rhs)
    q_cosine = solution[model.layout.n_state :]
    return q_cosine


def measurement_waveforms(time_s: np.ndarray, y_matrix: np.ndarray, model: NetworkModel) -> tuple[np.ndarray, np.ndarray]:
    """Return sending-end relay voltages and currents from simulated states."""

    del time_s
    currents = y_matrix[:, model.layout.source_current_slice]
    voltages = y_matrix[:, model.layout.sending_voltage_slice]
    return voltages, currents
