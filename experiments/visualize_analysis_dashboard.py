"""Create a pandas-based dashboard for model selection and feature analysis."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"

MODEL_CSV = EXPERIMENTS / "model_metrics_summary.csv"
SIGNIFICANCE_CSV = EXPERIMENTS / "feature_significance.csv"
ABLATION_CSV = EXPERIMENTS / "feature_ablation.csv"
IMPORTANCE_JSON = EXPERIMENTS / "feature_importance.json"
OUT_CSV = EXPERIMENTS / "feature_analysis_summary.csv"
OUT_PNG = EXPERIMENTS / "model_feature_analysis_dashboard.png"


def build_feature_summary():
    significance = pd.read_csv(SIGNIFICANCE_CSV)
    importance = json.loads(IMPORTANCE_JSON.read_text(encoding="utf-8"))
    importance_df = pd.DataFrame(
        importance["overall_all"].items(),
        columns=["feature", "rf_importance"],
    )
    statistical_df = (
        significance.groupby("feature", as_index=False)
        .agg(
            significant_label_count=("significant_fdr_0_05", "sum"),
            min_adjusted_p_value=("adjusted_p_value", "min"),
            max_abs_t_value=("t_value", lambda values: float(np.abs(values).max())),
        )
    )
    ablation = pd.read_csv(ABLATION_CSV)
    single_drop = ablation[
        ablation["dropped_features"].fillna("").str.match(r"^[^,]+$")
        & ablation["dropped_features"].notna()
    ][["dropped_features", "delta_r2", "delta_macro_f1"]].rename(
        columns={"dropped_features": "feature"}
    )

    result = statistical_df.merge(importance_df, on="feature", how="left")
    result = result.merge(single_drop, on="feature", how="left")
    result["rf_importance"] = result["rf_importance"].fillna(0.0)
    return result.sort_values(
        ["rf_importance", "significant_label_count"],
        ascending=[False, False],
    )


def main():
    models = pd.read_csv(MODEL_CSV).set_index("model_name")
    features = build_feature_summary()
    ablation = pd.read_csv(ABLATION_CSV)
    features.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    models[["r2_score", "macro_f1", "micro_f1"]].plot(
        kind="bar",
        ax=axes[0, 0],
        rot=0,
        ylim=(-0.2, 0.75),
        title="Model comparison: Random Forest selected",
    )
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 0].set_ylabel("Score")

    top = features.sort_values("rf_importance").tail(15)
    axes[0, 1].barh(top["feature"], top["rf_importance"], color="#4C72B0")
    axes[0, 1].set_title("Random Forest feature importance: top 15")
    axes[0, 1].set_xlabel("Mean importance")

    scatter = features.copy()
    scatter["minus_log10_adjusted_p"] = -np.log10(
        scatter["min_adjusted_p_value"].clip(lower=1e-300)
    )
    points = axes[1, 0].scatter(
        scatter["max_abs_t_value"],
        scatter["minus_log10_adjusted_p"],
        c=scatter["significant_label_count"],
        cmap="viridis",
        s=55,
    )
    for _, row in scatter.nlargest(8, "max_abs_t_value").iterrows():
        axes[1, 0].annotate(
            row["feature"],
            (row["max_abs_t_value"], row["minus_log10_adjusted_p"]),
            fontsize=8,
            xytext=(4, 3),
            textcoords="offset points",
        )
    axes[1, 0].axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=1)
    axes[1, 0].set_title("Welch t-test summary by numeric feature")
    axes[1, 0].set_xlabel("Maximum absolute t-value across labels")
    axes[1, 0].set_ylabel("-log10(min adjusted p-value)")
    fig.colorbar(points, ax=axes[1, 0], label="Significant label count")

    candidates = ablation[ablation["experiment"] != "baseline"].copy()
    axes[1, 1].scatter(candidates["delta_r2"], candidates["delta_macro_f1"], color="#DD8452")
    for _, row in candidates.iterrows():
        axes[1, 1].annotate(
            row["experiment"].replace("drop_", ""),
            (row["delta_r2"], row["delta_macro_f1"]),
            fontsize=8,
            xytext=(4, 3),
            textcoords="offset points",
        )
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].axvline(0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Feature ablation: removal impact")
    axes[1, 1].set_xlabel("Delta R2 after removal")
    axes[1, 1].set_ylabel("Delta macro F1 after removal")

    fig.suptitle(
        "Restaurant Recommendation Model Selection and Feature Diagnostics",
        fontsize=16,
    )
    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)
    print(f"saved: {OUT_CSV}")
    print(f"saved: {OUT_PNG}")
    print("\n=== feature analysis summary ===")
    print(features.to_string(index=False))


if __name__ == "__main__":
    main()
