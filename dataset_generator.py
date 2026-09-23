"""Incremental dataset generation for randomized fault scenarios."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from config import CONFIG, SimulationConfig
from fault_injection import sample_fault_scenario
from simulate import save_run_npz, simulate_scenario, validate_normal_run


def generate_dataset(
    n_runs: int,
    config: SimulationConfig = CONFIG,
    output_dir: str | Path | None = None,
    seed: int | None = None,
    validate_first: bool = True,
) -> Path:
    """Generate runs one at a time and write compressed NPZ files incrementally."""

    base_dir = Path(output_dir or Path(config.output_dir) / config.dataset_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = base_dir / config.metadata_filename
    rng = np.random.default_rng(config.random_seed if seed is None else seed)

    if validate_first and not validate_normal_run(config):
        raise RuntimeError("normal-run validation failed; refusing to generate dataset")

    fieldnames = [
        "run_id",
        "file",
        "fault_type",
        "faulted_phases",
        "fault_section",
        "fault_resistance_ohm",
        "fault_inception_angle_deg",
        "fault_duration_s",
        "t_start",
        "t_end",
        "load_p_total_w",
        "load_power_factor",
        "sampling_rate_hz",
        "source_ll_rms_v",
        "source_r_ohm",
        "source_x_ohm",
        "frequency_hz",
        "integrator",
        "fault_transition_s",
    ]
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run_id in range(n_runs):
            scenario = sample_fault_scenario(config, rng)
            result = simulate_scenario(scenario, config)
            filename = f"run_{run_id:06d}_{scenario.fault_type.lower()}.npz"
            save_run_npz(result, base_dir / filename)
            row = {"run_id": run_id, "file": filename}
            row.update(result["metadata"])
            writer.writerow({key: row.get(key, "") for key in fieldnames})
            if (run_id + 1) % 25 == 0 or run_id == n_runs - 1:
                print(f"generated {run_id + 1}/{n_runs} runs")
    print(f"Dataset metadata saved to {metadata_path}")
    return base_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate three-phase fault simulation dataset.")
    parser.add_argument("--runs", type=int, default=10, help="number of randomized simulations")
    parser.add_argument("--output-dir", type=str, default=None, help="dataset output directory")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--skip-validation", action="store_true", help="skip normal-run validation")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_dataset(
        args.runs,
        CONFIG,
        args.output_dir,
        args.seed,
        validate_first=not args.skip_validation,
    )
