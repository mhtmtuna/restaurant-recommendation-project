"""B1. 다중 모델 비교 — RF, XGBoost, CatBoost, LightGBM

OOF 5-fold + per-label threshold tuning 기준으로 비교.
결과: experiments/model_comparison.json, experiments/model_comparison.png
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_model import (
    CATEGORICAL_FEATURES,
    LABEL_COLUMNS,
    NUMERIC_FEATURES,
    multilabel_fold_indices,
    positive_class_probabilities,
    read_features,
    tune_thresholds,
    expand_seat_type,
)

N_FOLDS = 5
OUT_DIR = ROOT / "experiments"
OUT_JSON = OUT_DIR / "model_comparison.json"
OUT_PNG = OUT_DIR / "model_comparison.png"


def make_classifier(name):
    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            max_features=None,
            random_state=42,
            class_weight="balanced_subsample",
        )
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return MultiOutputClassifier(
            XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                scale_pos_weight=10,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
        )
    if name == "CatBoost":
        from catboost import CatBoostClassifier
        return MultiOutputClassifier(
            CatBoostClassifier(
                iterations=200,
                depth=4,
                learning_rate=0.1,
                auto_class_weights="Balanced",
                random_seed=42,
                verbose=0,
            )
        )
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return MultiOutputClassifier(
            LGBMClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                class_weight="balanced",
                random_state=42,
                verbose=-1,
            )
        )
    raise ValueError(f"Unknown model: {name}")


def make_pipeline(name, seat_columns):
    numeric_features = NUMERIC_FEATURES + seat_columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", make_classifier(name)),
    ])


def oof_scores_for(name, x_data, y_data, seat_columns):
    n_samples = len(x_data)
    n_labels = len(LABEL_COLUMNS)
    scores = np.zeros((n_samples, n_labels))

    for fold_idx, (train_idx, val_idx) in enumerate(multilabel_fold_indices(y_data, N_FOLDS), 1):
        print(f"  [{name}] fold {fold_idx}/{N_FOLDS}: train={len(train_idx)}, val={len(val_idx)}")
        pipe = make_pipeline(name, seat_columns)
        pipe.fit(x_data.iloc[train_idx], y_data.iloc[train_idx])
        scores[val_idx] = positive_class_probabilities(pipe, x_data.iloc[val_idx])

    return scores


def compute_metrics(y_true_arr, preds):
    results = {}
    f1s = []
    for i, label in enumerate(LABEL_COLUMNS):
        col_true = y_true_arr[:, i]
        col_pred = preds[:, i]
        tp = int((col_pred & col_true).sum())
        fp = int((col_pred & (1 - col_true)).sum())
        fn = int(((1 - col_pred) & col_true).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        results[label] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}
        f1s.append(f1)
    results["macro_f1"] = round(float(np.mean(f1s)), 4)
    return results


def plot_results(comparison):
    model_names = list(comparison.keys())
    macro_f1s = [comparison[m]["macro_f1"] for m in model_names]
    label_f1s = {
        label: [comparison[m]["per_label"][label]["f1"] for m in model_names]
        for label in LABEL_COLUMNS
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # macro F1 비교
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    bars = axes[0].bar(model_names, macro_f1s, color=colors)
    axes[0].set_title("Macro F1 (OOF, tuned threshold)", fontsize=13)
    axes[0].set_ylim(0, max(macro_f1s) * 1.3 + 0.05)
    axes[0].set_ylabel("Macro F1")
    for bar, val in zip(bars, macro_f1s):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    # 라벨별 F1 grouped bar
    x = np.arange(len(LABEL_COLUMNS))
    width = 0.2
    for i, (name, color) in enumerate(zip(model_names, colors)):
        vals = [label_f1s[label][i] for label in LABEL_COLUMNS]
        axes[1].bar(x + i * width, vals, width, label=name, color=color)
    axes[1].set_title("Per-label F1 (OOF, tuned threshold)", fontsize=13)
    axes[1].set_xticks(x + width * 1.5)
    short_labels = [l.replace("_", "\n") for l in LABEL_COLUMNS]
    axes[1].set_xticklabels(short_labels, fontsize=8)
    axes[1].set_ylabel("F1")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"saved: {OUT_PNG}")


def main():
    data = read_features()
    data, seat_columns = expand_seat_type(data)
    x_data = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES + seat_columns]
    y_data = data[LABEL_COLUMNS].astype(int)
    y_arr = y_data.to_numpy(dtype=int)

    model_names = ["RandomForest", "XGBoost", "CatBoost", "LightGBM"]
    comparison = {}

    for name in model_names:
        print(f"\n=== {name} ===")
        t0 = time.time()
        scores = oof_scores_for(name, x_data, y_data, seat_columns)
        elapsed = time.time() - t0

        thresholds = tune_thresholds(y_data, scores)
        preds = np.stack(
            [(scores[:, i] >= thresholds[label]["threshold"]).astype(int)
             for i, label in enumerate(LABEL_COLUMNS)],
            axis=1,
        )
        metrics = compute_metrics(y_arr, preds)

        comparison[name] = {
            "train_time_sec": round(elapsed, 1),
            "macro_f1": metrics["macro_f1"],
            "per_label": {label: metrics[label] for label in LABEL_COLUMNS},
            "optimal_thresholds": {label: thresholds[label]["threshold"] for label in LABEL_COLUMNS},
        }
        print(f"  macro F1: {metrics['macro_f1']:.4f}  ({elapsed:.1f}s)")
        for label in LABEL_COLUMNS:
            m = metrics[label]
            print(f"    {label:<20}  P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}  (thresh={thresholds[label]['threshold']:.2f})")

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"\nsaved: {OUT_JSON}")

    print("\n=== 최종 비교 ===")
    print(f"{'모델':<15} {'macro F1':>10} {'학습시간':>10}")
    print("-" * 38)
    for name in model_names:
        r = comparison[name]
        print(f"{name:<15} {r['macro_f1']:>10.4f} {r['train_time_sec']:>8.1f}s")

    plot_results(comparison)


if __name__ == "__main__":
    main()
