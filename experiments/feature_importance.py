"""B2. Feature Importance 분석

전체 / 라벨별 feature importance 시각화.
결과: experiments/feature_importance.png, experiments/feature_importance.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_training_common import (
    CATEGORICAL_FEATURES,
    LABEL_COLUMNS,
    NUMERIC_FEATURES,
    expand_seat_type,
    make_pipeline,
    read_features,
)
from train_model import estimator_factory

OUT_DIR = ROOT / "experiments"
OUT_PNG = OUT_DIR / "feature_importance.png"
OUT_JSON = OUT_DIR / "feature_importance.json"

FEATURE_CATEGORIES = {
    "metadata": ["rating", "review_count", "price", "photo_ratio", "collected_review_count"],
    "taste/value": ["taste_score", "taste_confidence", "taste_mentions",
                    "value_score", "value_confidence", "value_mentions",
                    "portion_score", "portion_confidence", "portion_mentions"],
    "atmosphere": ["brightness_score", "brightness_confidence", "brightness_mentions",
                   "noise_score", "noise_confidence", "noise_mentions",
                   "spaciousness_score", "spaciousness_confidence", "spaciousness_mentions"],
}


def get_feature_names(pipeline, seat_columns):
    preprocessor = pipeline.named_steps["preprocess"]
    numeric_features = NUMERIC_FEATURES + seat_columns
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return numeric_features + cat_names


def assign_category(name):
    for cat, members in FEATURE_CATEGORIES.items():
        if any(name.startswith(m) for m in members):
            return cat
    return "location/seat"


def main():
    data = read_features()
    data, seat_columns = expand_seat_type(data)
    x_data = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES + seat_columns]
    y_data = data[LABEL_COLUMNS].astype(int)

    print("fitting final model on all data...")
    pipe = make_pipeline(seat_columns, estimator_factory())
    pipe.fit(x_data, y_data)

    feature_names = get_feature_names(pipe, seat_columns)
    rf = pipe.named_steps["model"]

    # per-estimator importance (one RF per label output)
    importances_per_label = {}
    for i, label in enumerate(LABEL_COLUMNS):
        est = rf.estimators_[i] if hasattr(rf, "estimators_") else rf
        imp = est.feature_importances_
        importances_per_label[label] = dict(zip(feature_names, imp.tolist()))

    overall = np.mean(
        [list(importances_per_label[l].values()) for l in LABEL_COLUMNS], axis=0
    )
    overall_dict = dict(zip(feature_names, overall.tolist()))

    # top 15 전체
    top15_idx = np.argsort(overall)[-15:][::-1]
    top15_names = [feature_names[i] for i in top15_idx]
    top15_vals = [overall[i] for i in top15_idx]

    # 카테고리별 합산
    cat_totals = {}
    for name, val in overall_dict.items():
        cat = assign_category(name)
        cat_totals[cat] = cat_totals.get(cat, 0.0) + val
    total = sum(cat_totals.values()) or 1.0
    cat_pct = {k: round(v / total * 100, 1) for k, v in cat_totals.items()}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. 전체 top-15 bar
    colors_bar = ["#4C72B0" if not n.startswith("area") and not n.startswith("category")
                  else "#DD8452" for n in top15_names]
    axes[0].barh(top15_names[::-1], top15_vals[::-1], color=colors_bar[::-1])
    axes[0].set_title("Top-15 Feature Importance (overall)", fontsize=12)
    axes[0].set_xlabel("Mean importance")
    axes[0].tick_params(axis="y", labelsize=8)

    # 2. 카테고리별 파이
    pie_labels = [f"{k}\n{v}%" for k, v in cat_pct.items()]
    axes[1].pie(list(cat_pct.values()), labels=pie_labels, autopct="",
                colors=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    axes[1].set_title("Feature Category Contribution", fontsize=12)

    # 3. 라벨별 top-5 heatmap
    label_short = [l.replace("_", "\n") for l in LABEL_COLUMNS]
    top5_global = top15_names[:5]
    heatmap = np.array([
        [importances_per_label[label].get(f, 0.0) for f in top5_global]
        for label in LABEL_COLUMNS
    ])
    im = axes[2].imshow(heatmap, aspect="auto", cmap="YlOrRd")
    axes[2].set_xticks(range(len(top5_global)))
    axes[2].set_xticklabels(top5_global, rotation=30, ha="right", fontsize=8)
    axes[2].set_yticks(range(len(LABEL_COLUMNS)))
    axes[2].set_yticklabels(label_short, fontsize=8)
    axes[2].set_title("Per-label Importance (top-5 features)", fontsize=12)
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"saved: {OUT_PNG}")

    result = {
        "overall_top15": {n: round(v, 6) for n, v in zip(top15_names, top15_vals)},
        "overall_all": dict(
            sorted(
                ((name, round(value, 6)) for name, value in overall_dict.items()),
                key=lambda item: -item[1],
            )
        ),
        "category_contribution_pct": cat_pct,
        "per_label": {
            label: dict(sorted(importances_per_label[label].items(),
                               key=lambda x: -x[1])[:10])
            for label in LABEL_COLUMNS
        },
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"saved: {OUT_JSON}")

    print("\n=== 카테고리별 기여도 ===")
    for k, v in sorted(cat_pct.items(), key=lambda x: -x[1]):
        print(f"  {k:<15} {v:.1f}%")

    print("\n=== 전체 top-15 ===")
    for name, val in zip(top15_names, top15_vals):
        print(f"  {name:<35} {val:.4f}")


if __name__ == "__main__":
    main()
