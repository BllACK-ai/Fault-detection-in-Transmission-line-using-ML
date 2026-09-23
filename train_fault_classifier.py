"""Train and evaluate a Random Forest fault-type classifier."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib

os.environ.setdefault("MPLCONFIGDIR", str(Path("simulation_output") / "matplotlib_cache"))

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from feature_extraction import feature_columns


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


RANDOM_STATE = 12345
N_JOBS = 1
EXCLUDED_COLUMNS = {
    "fault_type",
    "faulted_phases",
    "fault_section",
    "fault_resistance_ohm",
    "fault_inception_angle_deg",
    "fault_duration_s",
}


def train_and_evaluate(
    dataset_path: str | Path = "fault_dataset.csv",
    model_path: str | Path = "fault_classifier_rf.joblib",
    confusion_matrix_path: str | Path = "confusion_matrix.png",
    feature_importance_path: str | Path = "feature_importances_top15.csv",
) -> dict[str, object]:
    dataset = _load_dataset(Path(dataset_path))
    features = list(feature_columns())
    _validate_dataset(dataset, features)
    _check_feature_leakage(features, dataset.columns)

    x = dataset[features].astype(float)
    y = dataset["fault_type"].astype(str)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"random_state: {RANDOM_STATE}")
    print(f"train shape: {x_train.shape[0]} rows x {x_train.shape[1]} features")
    print(f"test shape: {x_test.shape[0]} rows x {x_test.shape[1]} features")
    print(f"train class balance: {y_train.value_counts().sort_index().to_dict()}")
    print(f"test class balance: {y_test.value_counts().sort_index().to_dict()}")

    # Future tuning candidates: max_depth, min_samples_leaf, max_features,
    # class_weight, and bootstrap. Avoiding grid search keeps this first model
    # lightweight and reproducible on the target low-spec machine.
    classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, zero_division=0)
    labels = sorted(y.unique())
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    print(f"test accuracy: {accuracy:.6f}")
    print("classification report:")
    print(report_text)
    print("confusion matrix:")
    _print_confusion_matrix(matrix, labels)
    _save_confusion_matrix_plot(matrix, labels, Path(confusion_matrix_path))

    importances = _feature_importance_frame(classifier, features)
    top15 = importances.head(15)
    top15.to_csv(feature_importance_path, index=False)
    print("top 15 feature importances:")
    print(top15.to_string(index=False))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(classifier, x_train, y_train, cv=cv, scoring="accuracy", n_jobs=N_JOBS)
    print(f"5-fold stratified CV accuracy: {cv_scores.mean():.6f} +/- {cv_scores.std():.6f}")

    joblib.dump(classifier, model_path)
    print(f"saved model: {model_path}")
    print(f"saved confusion matrix image: {confusion_matrix_path}")
    print(f"saved top feature importances: {feature_importance_path}")

    _warn_if_suspiciously_perfect(accuracy, report_dict, features)
    weak_note = _weakest_class_note(report_dict)
    print("final summary:")
    print(f"  test accuracy: {accuracy:.6f}")
    print(f"  cross-val accuracy: {cv_scores.mean():.6f} +/- {cv_scores.std():.6f}")
    print(f"  weakest class note: {weak_note}")
    print(f"  leakage check: PASS; training matrix uses only the {len(features)} extracted feature columns")

    return {
        "accuracy": accuracy,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "confusion_matrix": matrix,
        "labels": labels,
        "top15": top15,
        "weak_note": weak_note,
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
    print(f"class balance: {dataset['fault_type'].value_counts().sort_index().to_dict()}")
    return dataset


def _validate_dataset(dataset: pd.DataFrame, features: list[str]) -> None:
    missing = [column for column in features + ["fault_type"] if column not in dataset.columns]
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")
    if dataset.shape[0] != 3000:
        print(f"WARNING: expected 3000 rows, found {dataset.shape[0]}")
    counts = dataset["fault_type"].value_counts()
    if len(counts) != 5 or not all(count == 600 for count in counts):
        print("WARNING: expected 5 classes with 600 rows each")
    feature_values = dataset[features].to_numpy(dtype=float)
    if not np.all(np.isfinite(feature_values)):
        raise ValueError("feature matrix contains NaN or infinite values")


def _check_feature_leakage(features: list[str], all_columns: Iterable[str]) -> None:
    leaked = sorted(EXCLUDED_COLUMNS.intersection(features))
    if leaked:
        raise ValueError(f"excluded leakage columns are present in features: {leaked}")
    unexpected = sorted(set(features) - set(all_columns))
    if unexpected:
        raise ValueError(f"feature list contains columns absent from dataset: {unexpected}")
    print(f"feature leakage check: PASS ({len(features)} features, excluded columns absent)")


def _print_confusion_matrix(matrix: np.ndarray, labels: list[str]) -> None:
    table = pd.DataFrame(matrix, index=[f"true_{label}" for label in labels], columns=[f"pred_{label}" for label in labels])
    print(table.to_string())


def _save_confusion_matrix_plot(matrix: np.ndarray, labels: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Predicted fault type")
    ax.set_ylabel("True fault type")
    ax.set_title("Random Forest Confusion Matrix")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="#111827")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _feature_importance_frame(classifier: RandomForestClassifier, features: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "feature": features,
            "importance": classifier.feature_importances_,
        }
    )
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def _warn_if_suspiciously_perfect(
    accuracy: float,
    report: dict[str, object],
    features: list[str],
) -> None:
    class_scores = [
        values
        for label, values in report.items()
        if isinstance(values, dict) and label not in {"macro avg", "weighted avg"}
    ]
    perfectish = accuracy >= 0.995 and all(float(scores.get("recall", 0.0)) >= 0.995 for scores in class_scores)
    if perfectish:
        print("WARNING: accuracy is suspiciously close to 100%.")
        print(f"  leakage columns excluded: {sorted(EXCLUDED_COLUMNS)}")
        print(f"  feature matrix column count: {len(features)}")
        print(f"  feature matrix columns: {features}")


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
        return f"no notably weak class; lowest is {weakest_label} with recall={recall:.3f}, F1={f1:.3f}"
    return f"{weakest_label} is notably weaker with recall={recall:.3f}, F1={f1:.3f}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Random Forest fault classifier.")
    parser.add_argument("--dataset", default="fault_dataset.csv", help="input feature CSV")
    parser.add_argument("--model", default="fault_classifier_rf.joblib", help="output joblib model path")
    parser.add_argument("--confusion-matrix", default="confusion_matrix.png", help="output PNG path")
    parser.add_argument("--feature-importances", default="feature_importances_top15.csv", help="output CSV path")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_and_evaluate(
        dataset_path=args.dataset,
        model_path=args.model,
        confusion_matrix_path=args.confusion_matrix,
        feature_importance_path=args.feature_importances,
    )
