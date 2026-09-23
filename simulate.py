"""Single-run simulation, validation, and demo generation."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from config import CONFIG, SimulationConfig
from fault_injection import (
    FAULT_TYPES,
    FaultScenario,
    fault_conductance_matrix,
    scenario_for_fault_type,
)
from network_model import (
    balanced_source_voltage,
    build_network_model,
    build_state_space_matrices,
    initial_state,
    matrix_state_derivative,
    measurement_waveforms,
)


def simulate_scenario(
    scenario: FaultScenario,
    config: SimulationConfig = CONFIG,
) -> dict[str, object]:
    """Run one time-domain simulation and return relay waveforms + metadata."""

    model = build_network_model(config, scenario.load_p_total_w, scenario.load_power_factor)
    dt = 1.0 / config.sampling_rate_hz
    time = np.arange(0.0, config.t_end_s + 0.5 * dt, dt)
    y0 = initial_state(config, model)
    fault_g = fault_conductance_matrix(scenario, config)
    matrix_pre = build_state_space_matrices(model)
    matrix_fault = (
        build_state_space_matrices(model, fault_g, scenario.fault_section)
        if fault_g is not None and scenario.fault_section is not None
        else matrix_pre
    )

    if config.integrator == "backward_euler":
        y = _integrate_backward_euler(time, y0, model, scenario, matrix_pre, matrix_fault)
    elif config.integrator == "trapezoidal":
        y = _integrate_trapezoidal(time, y0, model, scenario, matrix_pre, matrix_fault)
    elif config.integrator == "rk4":
        y = _integrate_rk4(time, y0, model, scenario, matrix_pre, matrix_fault)
    elif config.integrator in {"solve_ivp_radau", "solve_ivp_bdf"}:
        method = "Radau" if config.integrator == "solve_ivp_radau" else "BDF"
        y = _integrate_solve_ivp(time, y0, model, scenario, matrix_pre, matrix_fault, method, config)
    else:
        raise ValueError(f"unsupported integrator {config.integrator!r}")

    relay_v, relay_i = measurement_waveforms(time, y, model)
    return {
        "time": time,
        "Va": relay_v[:, 0],
        "Vb": relay_v[:, 1],
        "Vc": relay_v[:, 2],
        "Ia": relay_i[:, 0],
        "Ib": relay_i[:, 1],
        "Ic": relay_i[:, 2],
        "metadata": scenario.metadata()
        | {
            "sampling_rate_hz": config.sampling_rate_hz,
            "source_ll_rms_v": config.source_ll_rms_v,
            "source_r_ohm": config.source_r_ohm,
            "source_x_ohm": config.source_x_ohm,
            "frequency_hz": config.frequency_hz,
            "integrator": config.integrator,
            "fault_transition_s": config.fault_transition_s,
        },
    }


def save_run_npz(result: dict[str, object], output_path: str | Path) -> None:
    """Save one run as a compressed NPZ without retaining other runs in memory."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(result["metadata"], sort_keys=True)
    np.savez_compressed(
        path,
        time=result["time"],
        Va=result["Va"],
        Vb=result["Vb"],
        Vc=result["Vc"],
        Ia=result["Ia"],
        Ib=result["Ib"],
        Ic=result["Ic"],
        metadata_json=np.array(metadata_json),
    )


def validate_normal_run(config: SimulationConfig = CONFIG) -> bool:
    """Validate a healthy run for RMS, frequency, balance, and stability."""

    scenario = scenario_for_fault_type(config, "Normal")
    result = simulate_scenario(scenario, config)
    time = result["time"]
    start = int(0.1 * len(time))
    va = result["Va"][start:]
    vb = result["Vb"][start:]
    vc = result["Vc"][start:]
    currents = np.column_stack([result["Ia"], result["Ib"], result["Ic"]])

    target_phase_rms = config.source_ll_rms_v / math.sqrt(3.0)
    rms_values = np.array([_rms(va), _rms(vb), _rms(vc)])
    current_rms_values = np.array([_rms(currents[start:, phase]) for phase in range(3)])
    rms_ok = np.all(np.abs(rms_values - target_phase_rms) / target_phase_rms < 0.01)
    balance_ok = (np.max(rms_values) - np.min(rms_values)) / np.mean(rms_values) < 0.01
    freq = _zero_crossing_frequency(time[start:], va)
    freq_ok = abs(freq - config.frequency_hz) < 0.5
    finite_ok = all(np.all(np.isfinite(np.asarray(result[key]))) for key in ("Va", "Vb", "Vc", "Ia", "Ib", "Ic"))
    current_ok = np.all(np.isfinite(currents)) and np.max(np.abs(currents)) < 20_000.0

    checks = {
        "voltage_rms": rms_ok,
        "phase_balance": balance_ok,
        "frequency": freq_ok,
        "finite_waveforms": finite_ok,
        "current_reasonable": current_ok,
    }
    print("Validation summary")
    for name, ok in checks.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print(f"  measured phase RMS: {rms_values.round(2).tolist()} V")
    print(f"  measured current RMS: {current_rms_values.round(2).tolist()} A")
    print(f"  measured frequency: {freq:.3f} Hz")
    return all(checks.values())


def run_demo(config: SimulationConfig = CONFIG) -> None:
    """Run one scenario per fault type and save waveform files and SVG plots."""

    demo_dir = Path(config.output_dir) / config.demo_dir
    demo_dir.mkdir(parents=True, exist_ok=True)
    for index, fault_type in enumerate(FAULT_TYPES):
        scenario = scenario_for_fault_type(config, fault_type, index)
        result = simulate_scenario(scenario, config)
        save_run_npz(result, demo_dir / f"{fault_type.lower()}_demo.npz")
        save_demo_plot(result, demo_dir / f"{fault_type.lower()}_demo", f"{fault_type} demo")
        resistance = "None" if scenario.fault_resistance_ohm is None else f"{scenario.fault_resistance_ohm:.3f} ohm"
        print(
            f"{fault_type} demo: phases={scenario.faulted_phases or 'None'}, "
            f"section={scenario.fault_section or 'None'}, Rf={resistance}"
        )
    print(f"Demo waveforms saved to {demo_dir}")


def save_demo_plot(result: dict[str, object], output_stem: str | Path, title: str) -> None:
    """Save Va,Vb,Vc and Ia,Ib,Ic as SVG plus 300 DPI PNG demo plots."""

    os.environ.setdefault("MPLCONFIGDIR", str(Path("simulation_output") / "matplotlib_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["svg.fonttype"] = "none"
    time = np.asarray(result["time"])
    voltage = np.column_stack([result["Va"], result["Vb"], result["Vc"]])
    current = np.column_stack([result["Ia"], result["Ib"], result["Ic"]])
    colors = ("#b91c1c", "#2563eb", "#15803d")
    labels = ("A", "B", "C")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(title, fontsize=16, fontweight="bold", x=0.08, ha="left")
    for ax, data, ylabel in (
        (axes[0], voltage, "Sending-end voltage (V)"),
        (axes[1], current, "Sending-end current (A)"),
    ):
        for phase, label in enumerate(labels):
            ax.plot(time, data[:, phase], color=colors[phase], linewidth=1.25, label=f"Phase {label}")
        ax.set_ylabel(ylabel, fontsize=13, labelpad=14)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, color="#dbe3ec", linewidth=0.7)
        ax.margins(x=0.0, y=0.08)
    axes[1].set_xlabel(f"time (s), {time[0]:.3f} to {time[-1]:.3f}", fontsize=13, labelpad=10)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.985),
        ncol=3,
        frameon=True,
        fontsize=11,
    )
    fig.subplots_adjust(left=0.13, right=0.98, top=0.89, bottom=0.11, hspace=0.22)
    stem = Path(output_stem)
    fig.savefig(stem.with_suffix(".svg"), format="svg")
    fig.savefig(stem.with_suffix(".png"), format="png", dpi=300)
    plt.close(fig)


def _integrate_rk4(
    time: np.ndarray,
    y0: np.ndarray,
    model,
    scenario: FaultScenario,
    matrix_pre: tuple[np.ndarray, np.ndarray],
    matrix_fault: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    y = np.zeros((len(time), len(y0)), dtype=float)
    y[0] = y0
    pre_a, pre_b = matrix_pre
    fault_a, _ = matrix_fault
    delta_a = fault_a - pre_a

    for k in range(len(time) - 1):
        t = float(time[k])
        dt = float(time[k + 1] - time[k])
        scale1 = _fault_conductance_scale(t, scenario, model.config.fault_transition_s)
        scale2 = _fault_conductance_scale(t + 0.5 * dt, scenario, model.config.fault_transition_s)
        scale4 = _fault_conductance_scale(t + dt, scenario, model.config.fault_transition_s)
        a1 = pre_a + scale1 * delta_a
        a2 = pre_a + scale2 * delta_a
        a4 = pre_a + scale4 * delta_a
        k1 = matrix_state_derivative(t, y[k], model, a1, pre_b)
        k2 = matrix_state_derivative(t + 0.5 * dt, y[k] + 0.5 * dt * k1, model, a2, pre_b)
        k3 = matrix_state_derivative(t + 0.5 * dt, y[k] + 0.5 * dt * k2, model, a2, pre_b)
        k4 = matrix_state_derivative(t + dt, y[k] + dt * k3, model, a4, pre_b)
        y[k + 1] = y[k] + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return y


def _integrate_trapezoidal(
    time: np.ndarray,
    y0: np.ndarray,
    model,
    scenario: FaultScenario,
    matrix_pre: tuple[np.ndarray, np.ndarray],
    matrix_fault: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Integrate the linear time-varying-source model with an A-stable method.

    For each constant-topology interval, the state equation is
    y' = A*y + B*vs(t). The implicit trapezoidal rule solves

        (I - dt*A/2) y[k+1] =
            (I + dt*A/2) y[k] + dt*B*(vs[k] + vs[k+1])/2

    This is more robust than explicit RK4 for low-resistance faults because the
    shunt conductance can make the electrical time constant much smaller than
    the 10 kHz output sample period. Switching is still represented as a
    topology change: the configured fault transition is applied as a smooth
    conductance scale at the start and end of each implicit step.
    """

    y = np.zeros((len(time), len(y0)), dtype=float)
    y[0] = y0
    dt_nominal = float(time[1] - time[0]) if len(time) > 1 else 0.0
    pre_op = _trapezoid_operator(matrix_pre, dt_nominal)
    fault_op = _trapezoid_operator(matrix_fault, dt_nominal)
    source_values = np.vstack([balanced_source_voltage(float(t), model) for t in time])
    pre_a, pre_b = matrix_pre
    fault_a, _ = matrix_fault
    delta_a = fault_a - pre_a

    for k in range(len(time) - 1):
        t0 = float(time[k])
        t1 = float(time[k + 1])
        dt = t1 - t0
        scale_start = _fault_conductance_scale(t0, scenario, model.config.fault_transition_s)
        scale_end = _fault_conductance_scale(t1, scenario, model.config.fault_transition_s)
        if scale_start <= 0.0 and scale_end <= 0.0 and abs(dt - dt_nominal) < 1e-15:
            operator = pre_op
        elif scale_start >= 1.0 and scale_end >= 1.0 and abs(dt - dt_nominal) < 1e-15:
            operator = fault_op
        else:
            matrix_a_start = pre_a + scale_start * delta_a
            matrix_a_end = pre_a + scale_end * delta_a
            operator = _trapezoid_operator((matrix_a_start, pre_b), dt, matrix_a_end=matrix_a_end)
        source_sum = source_values[k] + source_values[k + 1]
        y[k + 1] = _trapezoid_step(y[k], source_sum, operator)
    return y


def _integrate_backward_euler(
    time: np.ndarray,
    y0: np.ndarray,
    model,
    scenario: FaultScenario,
    matrix_pre: tuple[np.ndarray, np.ndarray],
    matrix_fault: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Integrate y' = A*y + B*vs(t) with implicit backward Euler.

    Backward Euler is L-stable, so very fast LC modes introduced by small
    physical node capacitances and abrupt fault switching are damped instead of
    appearing as unrealistically large relay-voltage spikes.
    """

    y = np.zeros((len(time), len(y0)), dtype=float)
    y[0] = y0
    dt_nominal = float(time[1] - time[0]) if len(time) > 1 else 0.0
    pre_op = _backward_euler_operator(matrix_pre, dt_nominal)
    fault_op = _backward_euler_operator(matrix_fault, dt_nominal)
    source_values = np.vstack([balanced_source_voltage(float(t), model) for t in time])
    pre_a, pre_b = matrix_pre
    fault_a, _ = matrix_fault
    delta_a = fault_a - pre_a

    for k in range(len(time) - 1):
        t0 = float(time[k])
        t1 = float(time[k + 1])
        scale = _fault_conductance_scale(t1, scenario, model.config.fault_transition_s)
        if scale <= 0.0:
            operator = pre_op
        elif scale >= 1.0:
            operator = fault_op
        else:
            operator = _backward_euler_operator((pre_a + scale * delta_a, pre_b), dt_nominal)
        y[k + 1] = _backward_euler_step(y[k], source_values[k + 1], operator)
    return y


def _fault_conductance_scale(t_s: float, scenario: FaultScenario, transition_s: float) -> float:
    if scenario.fault_type == "Normal" or t_s < scenario.t_start or t_s >= scenario.t_end:
        return 0.0
    ramp = _fault_transition_ramp_duration(scenario, transition_s)
    if ramp <= 0.0:
        return 1.0
    if t_s < scenario.t_start + ramp:
        return _smoothstep((t_s - scenario.t_start) / ramp)
    if t_s > scenario.t_end - ramp:
        return _smoothstep((scenario.t_end - t_s) / ramp)
    return 1.0


def _fault_transition_ramp_duration(scenario: FaultScenario, transition_s: float) -> float:
    return min(max(transition_s, 0.0), max((scenario.t_end - scenario.t_start) * 0.5, 0.0))


def _smoothstep(value: float) -> float:
    x = min(max(value, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _trapezoid_operator(
    matrices: tuple[np.ndarray, np.ndarray],
    dt: float,
    matrix_a_end: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    matrix_a_start, matrix_b = matrices
    matrix_a_stop = matrix_a_start if matrix_a_end is None else matrix_a_end
    identity = np.eye(matrix_a_start.shape[0])
    left = identity - 0.5 * dt * matrix_a_stop
    right = identity + 0.5 * dt * matrix_a_start
    transition = np.linalg.solve(left, right)
    source_gain = np.linalg.solve(left, 0.5 * dt * matrix_b)
    return transition, source_gain


def _backward_euler_operator(
    matrices: tuple[np.ndarray, np.ndarray],
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    matrix_a, matrix_b = matrices
    identity = np.eye(matrix_a.shape[0])
    left = identity - dt * matrix_a
    transition = np.linalg.solve(left, identity)
    source_gain = np.linalg.solve(left, dt * matrix_b)
    return transition, source_gain


def _trapezoid_step(
    y_start: np.ndarray,
    source_sum: np.ndarray,
    operator: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    transition, source_gain = operator
    return transition @ y_start + source_gain @ source_sum


def _backward_euler_step(
    y_start: np.ndarray,
    source_next: np.ndarray,
    operator: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    transition, source_gain = operator
    return transition @ y_start + source_gain @ source_next


def _integrate_solve_ivp(
    time: np.ndarray,
    y0: np.ndarray,
    model,
    scenario: FaultScenario,
    matrix_pre: tuple[np.ndarray, np.ndarray],
    matrix_fault: tuple[np.ndarray, np.ndarray],
    method: str,
    config: SimulationConfig,
) -> np.ndarray:
    # Split integration at transition boundaries so adaptive solvers do not
    # step across abrupt changes when a zero-duration transition is requested.
    segments = [0.0, config.t_end_s]
    if scenario.fault_type != "Normal":
        ramp = _fault_transition_ramp_duration(scenario, config.fault_transition_s)
        segments.extend([scenario.t_start, scenario.t_end])
        if ramp > 0.0:
            segments.extend([scenario.t_start + ramp, scenario.t_end - ramp])
    segments = sorted(set(round(s, 12) for s in segments if 0.0 <= s <= config.t_end_s))
    outputs = []
    y_start = y0
    pre_a, pre_b = matrix_pre
    fault_a, _ = matrix_fault
    delta_a = fault_a - pre_a

    for idx in range(len(segments) - 1):
        a, b = segments[idx], segments[idx + 1]
        if b <= a:
            continue
        mask = (time >= a - 1e-12) & (time <= b + 1e-12)
        t_eval = time[mask]
        if idx > 0 and len(t_eval) > 0 and abs(t_eval[0] - a) < 1e-12:
            t_eval = t_eval[1:]
        if config.fault_transition_s <= 0.0:
            scale = _fault_conductance_scale((a + b) * 0.5, scenario, 0.0)
            matrix_a = pre_a + scale * delta_a

            def derivative(t_s: float, state: np.ndarray) -> np.ndarray:
                return matrix_state_derivative(t_s, state, model, matrix_a, pre_b)

        else:

            def derivative(t_s: float, state: np.ndarray) -> np.ndarray:
                scale = _fault_conductance_scale(t_s, scenario, config.fault_transition_s)
                matrix_a = pre_a + scale * delta_a
                return matrix_state_derivative(t_s, state, model, matrix_a, pre_b)

        solution = solve_ivp(
            derivative,
            (a, b),
            y_start,
            method=method,
            t_eval=t_eval if len(t_eval) else None,
            dense_output=True,
            rtol=config.rtol,
            atol=config.atol,
        )
        if not solution.success:
            raise RuntimeError(f"solve_ivp failed: {solution.message}")
        if len(t_eval):
            outputs.append(solution.y.T)
        y_start = solution.sol(b)
    if outputs:
        y = np.vstack(outputs)
    else:
        y = y0.reshape(1, -1)
    if len(y) != len(time):
        # Include the initial state if the first segment did not return it.
        if len(y) == len(time) - 1:
            y = np.vstack([y0, y])
        else:
            raise RuntimeError(f"internal time-grid mismatch: got {len(y)} states for {len(time)} samples")
    return y


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _zero_crossing_frequency(time: np.ndarray, values: np.ndarray) -> float:
    centered = values - np.mean(values)
    crossings = []
    for idx in range(1, len(centered)):
        if centered[idx - 1] < 0.0 <= centered[idx]:
            t0, t1 = time[idx - 1], time[idx]
            y0, y1 = centered[idx - 1], centered[idx]
            crossings.append(float(t0 - y0 * (t1 - t0) / (y1 - y0)))
    if len(crossings) < 2:
        return 0.0
    periods = np.diff(np.asarray(crossings))
    return float(1.0 / np.mean(periods))


if __name__ == "__main__":
    ok = validate_normal_run(CONFIG)
    run_demo(CONFIG)
    raise SystemExit(0 if ok else 1)
