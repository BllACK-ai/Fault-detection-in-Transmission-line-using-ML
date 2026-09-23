"""Analyze impedance-feature relationships with fault location."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


ANALYSIS_COLUMNS = (
    "Za_mag",
    "Zb_mag",
    "Zc_mag",
    "Z1_mag",
    "Za_angle_deg",
    "Zb_angle_deg",
    "Zc_angle_deg",
    "fault_resistance_ohm",
)
RANDOM_STATE = 12345


def analyze(dataset_path: str | Path = "fault_dataset.csv") -> pd.DataFrame:
    dataset = pd.read_csv(dataset_path)
    fault_rows = dataset[dataset["fault_type"].astype(str) != "Normal"].copy()
    fault_rows["fault_section"] = fault_rows["fault_section"].astype(int)
    fault_rows["fault_resistance_ohm"] = fault_rows["fault_resistance_ohm"].astype(float)

    x = fault_rows.loc[:, ANALYSIS_COLUMNS].astype(float)
    y = fault_rows["fault_section"].astype(int)
    mi = mutual_info_classif(
        x,
        y,
        discrete_features=False,
        random_state=RANDOM_STATE,
        n_neighbors=5,
    )

    rows = []
    for column, mi_value in zip(ANALYSIS_COLUMNS, mi):
        rows.append(
            {
                "feature": column,
                "pearson_corr_with_section": float(fault_rows[column].astype(float).corr(y, method="pearson")),
                "spearman_corr_with_section": float(fault_rows[column].astype(float).corr(y, method="spearman")),
                "mutual_info_with_section": float(mi_value),
            }
        )
    summary = pd.DataFrame(rows).sort_values("mutual_info_with_section", ascending=False).reset_index(drop=True)

    print(f"loaded dataset: {dataset_path}")
    print(f"fault-only shape: {fault_rows.shape[0]} rows x {fault_rows.shape[1]} columns")
    print(f"fault_section counts: {fault_rows['fault_section'].value_counts().sort_index().to_dict()}")
    print("section relationship metrics:")
    print(summary.to_string(index=False, formatters={column: "{:.6f}".format for column in summary.columns if column != "feature"}))

    grouped_resistance = fault_rows.groupby("fault_section")["fault_resistance_ohm"].agg(
        ["count", "mean", "median", "std", "min", "max"]
    )
    print("fault_resistance_ohm by section:")
    print(grouped_resistance.to_string(formatters={column: "{:.6f}".format for column in grouped_resistance.columns if column != "count"}))

    impedance_columns = [column for column in ANALYSIS_COLUMNS if column != "fault_resistance_ohm"]
    resistance_correlations = (
        fault_rows.loc[:, impedance_columns + ["fault_resistance_ohm"]]
        .astype(float)
        .corr(method="spearman")
        .loc[impedance_columns, "fault_resistance_ohm"]
        .abs()
        .sort_values(ascending=False)
    )
    print("absolute Spearman correlation with fault_resistance_ohm:")
    print(resistance_correlations.to_string(float_format="{:.6f}".format))

    output_path = Path("impedance_location_correlation_analysis.csv")
    summary.to_csv(output_path, index=False)
    print(f"saved analysis: {output_path}")
    return summary


if __name__ == "__main__":
    analyze()
