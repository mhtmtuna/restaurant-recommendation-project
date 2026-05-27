"""B3. Baseline 비교 — Random / Majority / Rating-only / Keyword-only vs 우리 모델

OOF 점수 없이 단순 규칙 기반 예측을 RF OOF 결과와 비교.
결과: experiments/baseline_comparison.json, experiments/baseline_comparison.png
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

from train_model import (
    LABEL_COLUMNS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    expand_seat_type,
    multilabel_fold_indices,
    make_pipeline,
    positive_class_probabilities,
    tune_thresholds,
    read_features,
)

OUT_DIR = ROOT / "experiments"
OUT_JSON = OUT_DIR / "baseline_comparison.json"
OUT_PNG = OUT_DIR / "baseline_comparison.png"
N_FOLDS = 5


def macro_f1(y_true, y_pred):
    f1s = []
    for i in range(y_true.shape[1]):
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]
        tp = int((col_pred & col_true).sum())
        fp = int((col_pred & (1 - col_true)).sum())
        fn = int(((1 - col_pred) & col_true).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)), f1s


def random_baseline(y_true):
    np.random.seed(42)
    pos_rates = y_true.mean(axis=0)
    preds = np.stack(
        [np.random.binomial(1, rate, size=len(y_true)) for rate in pos_rates], axis=1
    )
    return preds


def majority_baseline(y_true):
    # 항상 0 예측 (majority class)
    return np.zeros_like(y_true)


def rating_only_baseline(data, y_true, threshold=4.0):
    # rating >= threshold이면 모든 라벨 1 예측
    high_rating = (data["rating"].to_numpy() >= threshold).astype(int)
    return np.stack([high_rating] * y_true.shape[1], axis=1)


def keyword_only_baseline(data, y_true):
    # build_features.py가 이미 키워드 라벨을 계산해 놓음 — 그대로 사용
    return y_true.copy()


def rf_oof_baseline(x_data, y_data, seat_columns):
    n_samples = len(x_data)
    n_labels = len(LABEL_COLUMNS)
    scores = np.zeros((n_samples, n_labels))
    for _, (train_idx, val_idx) in enumerate(multilabel_fold_indices(y_data, N_FOLDS)):
        pipe = make_pipeline(seat_columns)
        pipe.fit(x_data.iloc[train_idx], y_data.iloc[train_idx])
        scores[val_idx] = positive_class_probabilities(pipe, x_data.iloc[val_idx])
    thresholds = tune_thresholds(y_data, scores)
    preds = np.stack(
        [(scores[:, i] >= thresholds[label]["threshold"]).astype(int)
         for i, label in enumerate(LABEL_COLUMNS)],
        axis=1,
    )
    return preds


def main():
    data = read_features()
    data, seat_columns = expand_seat_type(data)
    x_data = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES + seat_columns]
    y_data = data[LABEL_COLUMNS].astype(int)
    y_arr = y_data.to_numpy(dtype=int)

    print("computing baselines...")
    baselines = {
        "Random": random_baseline(y_arr),
        "Majority (all-zero)": majority_baseline(y_arr),
        "Rating-only (≥4.0)": rating_only_baseline(data, y_arr),
        "Keyword-only": keyword_only_baseline(y_arr, y_arr),
    }

    print("computing RF OOF (5-fold)...")
    baselines["RF + Threshold (ours)"] = rf_oof_baseline(x_data, y_data, seat_columns)

    results = {}
    print(f"\n{'모델':<28} {'macro F1':>10}  {'라벨별 F1'}")
    print("-" * 80)
    for name, preds in baselines.items():
        mf1, per_label_f1 = macro_f1(y_arr, preds)
        results[name] = {
            "macro_f1": round(mf1, 4),
            "per_label": {label: round(f1, 4) for label, f1 in zip(LABEL_COLUMNS, per_label_f1)},
        }
        label_str = "  ".join(f"{f:.2f}" for f in per_label_f1)
        print(f"  {name:<26} {mf1:>8.4f}  {label_str}")

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nsaved: {OUT_JSON}")

    # 시각화
    model_names = list(results.keys())
    macro_f1s = [results[m]["macro_f1"] for m in model_names]

    colors = ["#aaaaaa", "#aaaaaa", "#aaaaaa", "#aaaaaa", "#4C72B0"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bars = axes[0].bar(model_names, macro_f1s, color=colors)
    axes[0].set_title("Macro F1: Baselines vs Our Model", fontsize=13)
    axes[0].set_ylim(0, max(macro_f1s) * 1.35 + 0.02)
    axes[0].set_ylabel("Macro F1")
    axes[0].tick_params(axis="x", labelsize=8)
    for bar, val in zip(bars, macro_f1s):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    x = np.arange(len(LABEL_COLUMNS))
    width = 0.15
    for i, (name, color) in enumerate(zip(model_names, colors)):
        vals = [results[name]["per_label"][label] for label in LABEL_COLUMNS]
        axes[1].bar(x + i * width, vals, width, label=name, color=color,
                    alpha=1.0 if name == "RF + Threshold (ours)" else 0.6)
    axes[1].set_title("Per-label F1: Baselines vs Our Model", fontsize=13)
    axes[1].set_xticks(x + width * 2)
    axes[1].set_xticklabels([l.replace("_", "\n") for l in LABEL_COLUMNS], fontsize=8)
    axes[1].set_ylabel("F1")
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
