"""Train and evaluate a Random Forest fault-location classifier."""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import joblib

os.environ.setdefault("MPLCONFIGDIR", str(Path("simulation_output") / "matplotlib_cache"))

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from build_dataset import FULL_RUNS, _balanced_scenarios, _target_class_counts
from config import CONFIG
from feature_extraction import extract_features, feature_columns
from simulate import simulate_scenario
from train_fault_classifier import RANDOM_STATE


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_SNRS_DB = (40.0, 30.0, 20.0)
N_JOBS = 1
EXCLUDED_COLUMNS = {
    "fault_type",
    "faulted_phases",
    "fault_section",
    "fault_resistance_ohm",
    "fault_inception_angle_deg",
    "fault_duration_s",
}


def train_and_evaluate_location(
    dataset_path: str | Path = "fault_dataset.csv",
    model_path: str | Path = "fault_location_rf.joblib",
    confusion_matrix_path: str | Path = "confusion_matrix_location.png",
    feature_importance_path: str | Path = "feature_importances_location_top15.csv",
    noise_results_path: str | Path = "noise_robustness_location_results.csv",
    snrs_db: tuple[float, ...] = DEFAULT_SNRS_DB,
) -> dict[str, object]:
    dataset = _load_dataset(Path(dataset_path))
    features = list(feature_columns())
    _check_feature_leakage(features)

    fault_rows = dataset[dataset["fault_type"].astype(str) != "Normal"].copy()
    print("filtering: Normal rows excluded because fault_section is undefined for healthy runs")
    print(f"fault-only shape: {fault_rows.shape[0]} rows x {fault_rows.shape[1]} columns")
    print(f"fault_section distribution: {fault_rows['fault_section'].astype(int).value_counts().sort_index().to_dict()}")

    x = fault_rows[features].astype(float)
    y = fault_rows["fault_section"].astype(int)
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        x,
        y,
        fault_rows.index.to_numpy(),
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"random_state: {RANDOM_STATE}")
    print(f"train shape: {x_train.shape[0]} rows x {x_train.shape[1]} features")
    print(f"test shape: {x_test.shape[0]} rows x {x_test.shape[1]} features")
    print(f"train section balance: {y_train.value_counts().sort_index().to_dict()}")
    print(f"test section balance: {y_test.value_counts().sort_index().to_dict()}")

    classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    classifier.fit(x_train, y_train)

    labels = sorted(y.unique())
    y_pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, labels=labels, zero_division=0)
    report_dict = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    adjacency = _confusion_adjacency_summary(matrix, labels)

    print(f"test accuracy: {accuracy:.6f}")
    print("classification report:")
    print(report_text)
    print("confusion matrix:")
    _print_confusion_matrix(matrix, labels)
    print(
        "confusion pattern: "
        f"errors={adjacency['errors']}, adjacent={adjacency['adjacent_errors']}, "
        f"non_adjacent={adjacency['non_adjacent_errors']}, "
        f"adjacent_share={adjacency['adjacent_error_share']:.3f}"
    )
    _save_confusion_matrix_plot(matrix, labels, Path(confusion_matrix_path))

    importances = _feature_importance_frame(classifier, features)
    top15 = importances.head(15)
    top15.to_csv(feature_importance_path, index=False)
    print("top 15 location feature importances:")
    print(top15.to_string(index=False))
    comparison_note = _compare_to_fault_type_importances(top15)
    print(f"feature-importance comparison: {comparison_note}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(classifier, x_train, y_train, cv=cv, scoring="accuracy", n_jobs=N_JOBS)
    print(f"5-fold stratified CV accuracy: {cv_scores.mean():.6f} +/- {cv_scores.std():.6f}")

    joblib.dump(classifier, model_path)
    print(f"saved model: {model_path}")
    print(f"saved confusion matrix image: {confusion_matrix_path}")
    print(f"saved top feature importances: {feature_importance_path}")

    noise_summary = _evaluate_noise_robustness(classifier, dataset, idx_test, y_test.reset_index(drop=True), labels, snrs_db)
    noise_summary.to_csv(noise_results_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print("location noise robustness results:")
    print(noise_summary.to_string(index=False, formatters={"test_accuracy": "{:.6f}".format}))
    print(f"saved location noise results: {noise_results_path}")

    weak_note = _weakest_class_note(report_dict)
    difficulty_note = _difficulty_note(accuracy, float(noise_summary.loc[noise_summary["snr_db"] == "20 dB", "test_accuracy"].iloc[0]))
    print("final summary:")
    print(f"  test accuracy: {accuracy:.6f}")
    print(f"  cross-val accuracy: {cv_scores.mean():.6f} +/- {cv_scores.std():.6f}")
    print(
        "  confusion matrix pattern: "
        f"{adjacency['adjacent_errors']} adjacent-section errors vs "
        f"{adjacency['non_adjacent_errors']} non-adjacent errors"
    )
    print("  noise robustness:")
    print(noise_summary.to_string(index=False, formatters={"test_accuracy": "{:.6f}".format}))
    print(f"  weakest section note: {weak_note}")
    print(f"  location-vs-type note: {difficulty_note}")
    print(f"  leakage check: PASS; training matrix uses only the {len(features)} extracted feature columns")

    return {
        "accuracy": accuracy,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "confusion_matrix": matrix,
        "adjacency": adjacency,
        "top15": top15,
        "noise_summary": noise_summary,
    }


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
    print(f"fault_type balance: {dataset['fault_type'].value_counts().sort_index().to_dict()}")
    return dataset


def _check_feature_leakage(features: list[str]) -> None:
    leaked = sorted(set(features).intersection(EXCLUDED_COLUMNS))
    if leaked:
        raise ValueError(f"excluded leakage columns are present in features: {leaked}")
    print(f"feature leakage check: PASS ({len(features)} features, excluded columns absent)")


def _print_confusion_matrix(matrix: np.ndarray, labels: list[int]) -> None:
    table = pd.DataFrame(matrix, index=[f"true_{label}" for label in labels], columns=[f"pred_{label}" for label in labels])
    print(table.to_string())


def _save_confusion_matrix_plot(matrix: np.ndarray, labels: list[int], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)), labels=[str(label) for label in labels])
    ax.set_yticks(np.arange(len(labels)), labels=[str(label) for label in labels])
    ax.set_xlabel("Predicted fault section")
    ax.set_ylabel("True fault section")
    ax.set_title("Random Forest Fault-Location Confusion Matrix")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="#111827")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _feature_importance_frame(classifier: RandomForestClassifier, features: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame({"feature": features, "importance": classifier.feature_importances_})
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def _compare_to_fault_type_importances(location_top15: pd.DataFrame) -> str:
    path = Path("feature_importances_top15.csv")
    location_features = set(location_top15["feature"].astype(str))
    if not path.exists():
        return "fault-type top-feature file not found; no direct comparison made"
    type_top15 = pd.read_csv(path)
    type_features = set(type_top15["feature"].astype(str))
    overlap = sorted(location_features.intersection(type_features))
    seq_location = sum(_is_sequence_feature(feature) for feature in location_features)
    seq_type = sum(_is_sequence_feature(feature) for feature in type_features)
    return (
        f"{len(overlap)}/15 overlap with fault-type top features; "
        f"sequence features in location top15={seq_location}, fault-type top15={seq_type}; "
        f"overlap={overlap}"
    )


def _is_sequence_feature(feature: str) -> bool:
    return feature in {"V0_rms", "V1_rms", "V2_rms", "I0_rms", "I1_rms", "I2_rms", "V2_over_V1", "V0_over_V1"}


def _confusion_adjacency_summary(matrix: np.ndarray, labels: list[int]) -> dict[str, float]:
    errors = 0
    adjacent = 0
    non_adjacent = 0
    for true_index, true_label in enumerate(labels):
        for pred_index, pred_label in enumerate(labels):
            count = int(matrix[true_index, pred_index])
            if true_label == pred_label:
                continue
            errors += count
            if abs(int(true_label) - int(pred_label)) == 1:
                adjacent += count
            else:
                non_adjacent += count
    return {
        "errors": errors,
        "adjacent_errors": adjacent,
        "non_adjacent_errors": non_adjacent,
        "adjacent_error_share": float(adjacent / errors) if errors else 0.0,
    }


def _evaluate_noise_robustness(
    classifier: RandomForestClassifier,
    dataset: pd.DataFrame,
    test_indices: np.ndarray,
    y_true: pd.Series,
    labels: list[int],
    snrs_db: tuple[float, ...],
) -> pd.DataFrame:
    scenarios = _regenerate_scenarios(len(dataset))
    _check_scenario_alignment(dataset, scenarios)
    rows_by_level: dict[str, list[dict[str, object]]] = {"clean": []}
    rows_by_level.update({f"{snr:g} dB": [] for snr in snrs_db})
    rng = np.random.default_rng(20260718)
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
                f"processed {count}/{len(test_indices)} location test waveforms; "
                f"elapsed={_format_seconds(elapsed)}; eta={_format_seconds(eta)}"
            )

    features = list(feature_columns())
    summary_rows = []
    for level, rows in rows_by_level.items():
        x_noisy = pd.DataFrame(rows, columns=features).astype(float)
        predictions = classifier.predict(x_noisy)
        matrix = confusion_matrix(y_true, predictions, labels=labels)
        summary_rows.append(
            {
                "snr_db": level,
                "test_accuracy": accuracy_score(y_true, predictions),
                "correct": int(np.trace(matrix)),
                "total": int(matrix.sum()),
            }
        )
    return pd.DataFrame(summary_rows)


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


def _weakest_class_note(report: dict[str, object]) -> str:
    class_rows = {
        label: values
        for label, values in report.items()
        if isinstance(values, dict) and label not in {"macro avg", "weighted avg"}
    }
    weakest_label, weakest_values = min(
        class_rows.items(),
        key=lambda item: (float(item[1].get("recall", 0.0)), float(item[1].get("f1-score", 0.0))),
    )
    recall = float(weakest_values.get("recall", 0.0))
    f1 = float(weakest_values.get("f1-score", 0.0))
    if recall >= 0.95 and f1 >= 0.95:
        return f"no notably weak section; lowest is {weakest_label} with recall={recall:.3f}, F1={f1:.3f}"
    return f"section {weakest_label} is weaker with recall={recall:.3f}, F1={f1:.3f}"


def _difficulty_note(clean_location_accuracy: float, noisy_20db_location_accuracy: float) -> str:
    fault_type_clean = 1.0
    fault_type_20db = 0.7288888888888889
    if clean_location_accuracy < fault_type_clean or noisy_20db_location_accuracy < fault_type_20db:
        return (
            "location classification is harder in this dataset, especially under noise, "
            "because distance-to-fault signatures are subtler than fault-type symmetry signatures"
        )
    return (
        "location classification is not harder on these measured metrics; "
        "it matches or exceeds the previous fault-type robustness numbers"
    )


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
    parser = argparse.ArgumentParser(description="Train Random Forest fault-location classifier.")
    parser.add_argument("--dataset", default="fault_dataset.csv", help="input feature CSV")
    parser.add_argument("--model", default="fault_location_rf.joblib", help="output joblib model path")
    parser.add_argument("--confusion-matrix", default="confusion_matrix_location.png", help="output PNG path")
    parser.add_argument("--feature-importances", default="feature_importances_location_top15.csv", help="output CSV path")
    parser.add_argument("--noise-output", default="noise_robustness_location_results.csv", help="output noise summary CSV")
    parser.add_argument("--snr-db", nargs="*", type=float, default=list(DEFAULT_SNRS_DB), help="SNR levels in dB")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_and_evaluate_location(
        dataset_path=args.dataset,
        model_path=args.model,
        confusion_matrix_path=args.confusion_matrix,
        feature_importance_path=args.feature_importances,
        noise_results_path=args.noise_output,
        snrs_db=tuple(args.snr_db),
    )
