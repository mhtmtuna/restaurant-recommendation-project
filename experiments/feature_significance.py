"""Analyze numeric feature associations with each recommendation label.

Uses Welch's t-test and Benjamini-Hochberg adjusted p-values. The test results
describe association, not causal importance. Combine them with model feature
importance before removing features.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_training_common import LABEL_COLUMNS, NUMERIC_FEATURES, read_features

OUT_CSV = ROOT / "experiments" / "feature_significance.csv"
OUT_JSON = ROOT / "experiments" / "feature_significance.json"


def adjust_bh(p_values):
    """Return Benjamini-Hochberg adjusted p-values in original order."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def analyze():
    data = read_features()
    rows = []

    for feature in NUMERIC_FEATURES:
        values = pd.to_numeric(data[feature], errors="coerce")
        for label in LABEL_COLUMNS:
            positive = values[data[label] == 1].dropna()
            negative = values[data[label] == 0].dropna()
            if values.nunique() <= 1 or positive.empty or negative.empty:
                t_value, p_value = 0.0, 1.0
            else:
                t_value, p_value = ttest_ind(
                    positive,
                    negative,
                    equal_var=False,
                    nan_policy="omit",
                )
                if np.isnan(t_value) or np.isnan(p_value):
                    t_value, p_value = 0.0, 1.0

            rows.append(
                {
                    "feature": feature,
                    "label": label,
                    "positive_mean": float(positive.mean()) if not positive.empty else None,
                    "negative_mean": float(negative.mean()) if not negative.empty else None,
                    "mean_difference": float(positive.mean() - negative.mean())
                    if not positive.empty and not negative.empty
                    else None,
                    "t_value": float(t_value),
                    "p_value": float(p_value),
                    "feature_std": float(values.std()) if values.notna().any() else 0.0,
                }
            )

    result = pd.DataFrame(rows)
    result["adjusted_p_value"] = adjust_bh(result["p_value"])
    result["significant_fdr_0_05"] = result["adjusted_p_value"] < 0.05
    result.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    summary = (
        result.groupby("feature", as_index=False)
        .agg(
            feature_std=("feature_std", "first"),
            significant_label_count=("significant_fdr_0_05", "sum"),
            min_adjusted_p_value=("adjusted_p_value", "min"),
            max_abs_t_value=("t_value", lambda values: float(np.abs(values).max())),
        )
        .sort_values(
            ["significant_label_count", "max_abs_t_value"],
            ascending=[False, False],
        )
    )
    OUT_JSON.write_text(
        json.dumps(
            {
                "method": "Welch t-test per numeric feature and label with Benjamini-Hochberg FDR correction",
                "total_restaurants": len(data),
                "summary": summary.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"saved: {OUT_CSV}")
    print(f"saved: {OUT_JSON}")
    print("\n=== numeric feature significance summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    analyze()
