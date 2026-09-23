"""Evaluate saved fault classifier under raw waveform measurement noise."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from build_dataset import FULL_RUNS, _balanced_scenarios, _target_class_counts
from config import CONFIG
from feature_extraction import extract_features, feature_columns
from simulate import simulate_scenario
from train_fault_classifier import RANDOM_STATE


DEFAULT_SNRS_DB = (40.0, 30.0, 20.0)


def evaluate_noise_robustness(
    dataset_path: str | Path = "fault_dataset.csv",
    model_path: str | Path = "fault_classifier_rf.joblib",
    output_path: str | Path = "noise_robustness_results.csv",
    snrs_db: tuple[float, ...] = DEFAULT_SNRS_DB,
    seed: int = 20260718,
) -> pd.DataFrame:
    """Regenerate held-out waveforms, add noise, and evaluate saved model.

    Noise is injected into the raw relay voltage/current channels before
    feature extraction. For each channel, Gaussian noise power is set from that
    channel's own signal power and the requested SNR, which approximates
    independent sensor/PMU measurement noise without leaking labels or scenario
    parameters into the feature matrix.
    """

    dataset = _load_dataset(Path(dataset_path))
    model = joblib.load(model_path)
    features = list(feature_columns())
    _validate_feature_columns(features)

    test_indices = _test_indices(dataset)
    scenarios = _regenerate_scenarios(len(dataset))
    _check_scenario_alignment(dataset, scenarios)
    y_true = dataset.iloc[test_indices]["fault_type"].astype(str).reset_index(drop=True)
    labels = sorted(dataset["fault_type"].astype(str).unique())

    rows_by_level: dict[str, list[dict[str, object]]] = {"clean": []}
    rows_by_level.update({f"{snr:g} dB": [] for snr in snrs_db})
    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    for count, row_index in enumerate(test_indices, start=1):
        scenario = scenarios[int(row_index)]
        result = simulate_scenario(scenario, CONFIG)
        rows_by_level["clean"].append(_extract_result_features(result))
        for snr in snrs_db:
            noisy = _with_measurement_noise(result, snr, rng)
            rows_by_level[f"{snr:g} dB"].append(_extract_result_features(noisy))
        if count % 100 == 0 or count == len(test_indices):
            elapsed = time.perf_counter() - start
            eta = elapsed / count * (len(test_indices) - count)
            print(
                f"processed {count}/{len(test_indices)} test waveforms; "
                f"elapsed={_format_seconds(elapsed)}; eta={_format_seconds(eta)}"
            )

    summary_rows = []
    for level, rows in rows_by_level.items():
        x_noisy = pd.DataFrame(rows, columns=features).astype(float)
        predictions = model.predict(x_noisy)
        accuracy = accuracy_score(y_true, predictions)
        matrix = confusion_matrix(y_true, predictions, labels=labels)
        summary_rows.append(
            {
                "snr_db": level,
                "test_accuracy": accuracy,
                "correct": int(np.trace(matrix)),
                "total": int(matrix.sum()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print("noise robustness results:")
    print(summary.to_string(index=False, formatters={"test_accuracy": "{:.6f}".format}))
    print(f"saved results: {output_path}")
    return summary


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        fallback = Path("simulation_output") / path.name
        if fallback.exists():
            path = fallback
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    dataset = pd.read_csv(path)
    print(f"loaded dataset: {path}")
    print(f"dataset shape: {dataset.shape[0]} rows x {dataset.shape[1]} columns")
    print(f"class balance: {dataset['fault_type'].value_counts().sort_index().to_dict()}")
    return dataset


def _validate_feature_columns(features: list[str]) -> None:
    excluded = {
        "fault_type",
        "faulted_phases",
        "fault_section",
        "fault_resistance_ohm",
        "fault_inception_angle_deg",
        "fault_duration_s",
    }
    leaked = sorted(set(features).intersection(excluded))
    if leaked:
        raise ValueError(f"leakage columns in feature list: {leaked}")
    print(f"leakage check: PASS ({len(features)} extracted features only)")


def _test_indices(dataset: pd.DataFrame) -> np.ndarray:
    indices = np.arange(len(dataset))
    _, test_indices = train_test_split(
        indices,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=dataset["fault_type"].astype(str),
    )
    print(f"recreated test split with random_state={RANDOM_STATE}: {len(test_indices)} rows")
    return test_indices


def _regenerate_scenarios(n_rows: int):
    if n_rows != FULL_RUNS:
        print(f"WARNING: expected {FULL_RUNS} generated rows, found {n_rows}")
    rng = np.random.default_rng(CONFIG.random_seed)
    return list(_balanced_scenarios(CONFIG, rng, _target_class_counts(n_rows)))


def _check_scenario_alignment(dataset: pd.DataFrame, scenarios) -> None:
    generated_labels = [scenario.fault_type for scenario in scenarios]
    csv_labels = dataset["fault_type"].astype(str).tolist()
    mismatches = sum(left != right for left, right in zip(generated_labels, csv_labels))
    if mismatches:
        raise RuntimeError(f"regenerated scenario order does not match dataset labels; mismatches={mismatches}")
    print("scenario regeneration check: PASS; regenerated labels match dataset row order")


def _extract_result_features(result: dict[str, object]) -> dict[str, object]:
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
    return {column: row[column] for column in feature_columns()}


def _with_measurement_noise(
    result: dict[str, object],
    snr_db: float,
    rng: np.random.Generator,
) -> dict[str, object]:
    noisy = dict(result)
    for channel in ("Va", "Vb", "Vc", "Ia", "Ib", "Ic"):
        signal = np.asarray(result[channel], dtype=float)
        signal_power = float(np.mean(signal**2))
        noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
        noise_std = float(np.sqrt(max(noise_power, 0.0)))
        noisy[channel] = signal + rng.normal(0.0, noise_std, size=signal.shape)
    return noisy


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
    parser = argparse.ArgumentParser(description="Evaluate noise robustness of saved fault classifier.")
    parser.add_argument("--dataset", default="fault_dataset.csv", help="input feature CSV used for split labels")
    parser.add_argument("--model", default="fault_classifier_rf.joblib", help="trained Random Forest joblib path")
    parser.add_argument("--output", default="noise_robustness_results.csv", help="output summary CSV")
    parser.add_argument("--snr-db", nargs="*", type=float, default=list(DEFAULT_SNRS_DB), help="SNR levels in dB")
    parser.add_argument("--seed", type=int, default=20260718, help="measurement-noise random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_noise_robustness(
        dataset_path=args.dataset,
        model_path=args.model,
        output_path=args.output,
        snrs_db=tuple(args.snr_db),
        seed=args.seed,
    )
