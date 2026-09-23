"""Build an incremental feature CSV for the fault diagnosis ML pipeline."""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from config import CONFIG, SimulationConfig
from fault_injection import FAULT_TYPES, FaultScenario, sample_fault_scenario
from feature_extraction import extract_features, feature_columns, output_columns
from simulate import simulate_scenario


QUICK_RUNS = 20
FULL_RUNS = 3000
PROGRESS_EVERY = 100


def build_feature_dataset(
    n_runs: int,
    output_path: str | Path,
    config: SimulationConfig = CONFIG,
    seed: int | None = None,
    progress_every: int = PROGRESS_EVERY,
    collect_rows: bool = False,
) -> dict[str, object]:
    """Simulate balanced scenarios, extract features, and append rows to CSV.

    Rows are written immediately as each simulation finishes, so an interrupted
    long run still leaves a usable partial CSV instead of losing all progress.
    Class balancing is achieved by accepting scenarios from the existing
    sampler only while that fault type still has remaining quota.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.random_seed if seed is None else seed)
    target_counts = _target_class_counts(n_runs)
    counts: Counter[str] = Counter()
    bad_feature_rows = 0
    rows_for_summary: list[dict[str, object]] = []
    start_time = time.perf_counter()
    fields = list(output_columns())

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run_id, scenario in enumerate(_balanced_scenarios(config, rng, target_counts), start=1):
            result = simulate_scenario(scenario, config)
            row = extract_features(
                result["time"],
                result["Va"],
                result["Vb"],
                result["Vc"],
                result["Ia"],
                result["Ib"],
                result["Ic"],
                result["metadata"],
                fs=float(result["metadata"]["sampling_rate_hz"]),
            )
            if not _feature_values_are_finite(row):
                bad_feature_rows += 1
            writer.writerow({column: row.get(column, "") for column in fields})
            handle.flush()
            counts[scenario.fault_type] += 1
            if collect_rows:
                rows_for_summary.append(row)
            if run_id % progress_every == 0 or run_id == n_runs:
                elapsed = time.perf_counter() - start_time
                eta = elapsed / run_id * (n_runs - run_id)
                print(
                    f"generated {run_id}/{n_runs} rows; "
                    f"elapsed={_format_seconds(elapsed)}; eta={_format_seconds(eta)}; "
                    f"class_balance={dict(counts)}"
                )

    elapsed = time.perf_counter() - start_time
    summary = {
        "output_path": path,
        "rows": n_runs,
        "columns": len(fields),
        "feature_columns": list(feature_columns()),
        "output_columns": fields,
        "class_balance": dict(counts),
        "nonfinite_feature_rows": bad_feature_rows,
        "elapsed_s": elapsed,
        "rows_for_summary": rows_for_summary,
    }
    _print_final_summary(summary)
    return summary


def quick_test(
    config: SimulationConfig = CONFIG,
    seed: int | None = None,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    path = Path(output_path or Path(config.output_dir) / "quick_fault_dataset.csv")
    summary = build_feature_dataset(
        QUICK_RUNS,
        path,
        config=config,
        seed=seed,
        progress_every=QUICK_RUNS,
        collect_rows=True,
    )
    rows = summary["rows_for_summary"]
    if not isinstance(rows, list) or len(rows) != QUICK_RUNS:
        raise RuntimeError("quick test did not collect the expected rows")
    if len(summary["feature_columns"]) != 51:
        raise RuntimeError(f"expected 51 feature columns, got {len(summary['feature_columns'])}")
    if len(summary["output_columns"]) != 57:
        raise RuntimeError(f"expected 57 output columns, got {len(summary['output_columns'])}")
    if int(summary["nonfinite_feature_rows"]) != 0:
        raise RuntimeError("quick test found non-finite feature values")
    _print_quick_preview(rows, summary)
    return summary


def _balanced_scenarios(
    config: SimulationConfig,
    rng: np.random.Generator,
    target_counts: dict[str, int],
) -> Iterable[FaultScenario]:
    counts: Counter[str] = Counter()
    total_target = sum(target_counts.values())
    while sum(counts.values()) < total_target:
        scenario = sample_fault_scenario(config, rng)
        if counts[scenario.fault_type] >= target_counts[scenario.fault_type]:
            continue
        counts[scenario.fault_type] += 1
        yield scenario


def _target_class_counts(n_runs: int) -> dict[str, int]:
    base = n_runs // len(FAULT_TYPES)
    remainder = n_runs % len(FAULT_TYPES)
    return {
        fault_type: base + (1 if index < remainder else 0)
        for index, fault_type in enumerate(FAULT_TYPES)
    }


def _feature_values_are_finite(row: dict[str, object]) -> bool:
    values = np.array([float(row[column]) for column in feature_columns()], dtype=float)
    return bool(np.all(np.isfinite(values)))


def _print_final_summary(summary: dict[str, object]) -> None:
    nonfinite = int(summary["nonfinite_feature_rows"])
    print(f"saved feature dataset: {summary['output_path']}")
    print(f"final shape: {summary['rows']} rows x {summary['columns']} columns")
    print(f"class balance: {summary['class_balance']}")
    if nonfinite:
        print(f"WARNING: found {nonfinite} rows with NaN or infinite feature values")
    else:
        print("NaN/infinite feature values: 0")
    print(f"total wall-clock time: {_format_seconds(float(summary['elapsed_s']))}")


def _print_quick_preview(rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    elapsed = float(summary["elapsed_s"])
    estimate = elapsed / max(len(rows), 1) * FULL_RUNS
    print("column names:")
    print(", ".join(str(column) for column in summary["output_columns"]))
    print("example row:")
    _print_row_table(rows[0], summary["output_columns"])
    print("quick-test finite-value check: PASS")
    print(f"estimated 3000-run time from quick test: {_format_seconds(estimate)}")
    _try_print_pandas_shape(rows)


def _print_row_table(row: dict[str, object], columns: list[str]) -> None:
    for column in columns:
        value = row.get(column, "")
        if isinstance(value, float):
            print(f"  {column}: {value:.6g}")
        else:
            print(f"  {column}: {value}")


def _try_print_pandas_shape(rows: list[dict[str, object]]) -> None:
    try:
        import pandas as pd
    except ModuleNotFoundError:
        print("pandas is not installed; quick preview used standard-library CSV rows")
        return
    frame = pd.DataFrame(rows)
    print(f"pandas dataframe shape: {frame.shape[0]} rows x {frame.shape[1]} columns")


def _format_seconds(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    minutes, sec = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fault diagnosis feature dataset.")
    parser.add_argument("--full", action="store_true", help="after the 20-run quick test, build the full dataset")
    parser.add_argument("--runs", type=int, default=FULL_RUNS, help="number of rows for --full mode")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--quick-output", type=str, default=None, help="CSV path for the 20-run quick test")
    parser.add_argument("--output", type=str, default="fault_dataset.csv", help="CSV path for the full dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    quick_test(CONFIG, seed=args.seed, output_path=args.quick_output)
    if args.full:
        build_feature_dataset(args.runs, args.output, CONFIG, seed=args.seed)
    else:
        print("Full 3000-run generation was not started. Re-run with --full to generate it.")
