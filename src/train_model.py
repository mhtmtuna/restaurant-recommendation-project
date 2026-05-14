import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "restaurants_features.csv"
MODEL_PATH = ROOT / "models" / "restaurant_recommender.joblib"
REPORT_PATH = ROOT / "data" / "model_report.json"
PREDICTIONS_PATH = ROOT / "data" / "restaurant_label_scores.csv"

LABEL_COLUMNS = [
    "couple_meal",
    "couple_drink",
    "friend_meal",
    "friend_drink",
    "business_meal",
    "business_drink",
]

NUMERIC_FEATURES = [
    "rating",
    "review_count",
    "price",
    "photo_ratio",
    "collected_review_count",
    "taste_score",
    "taste_confidence",
    "taste_mentions",
    "value_score",
    "value_confidence",
    "value_mentions",
    "portion_score",
    "portion_confidence",
    "portion_mentions",
    "brightness_score",
    "brightness_confidence",
    "brightness_mentions",
    "noise_score",
    "noise_confidence",
    "noise_mentions",
    "spaciousness_score",
    "spaciousness_confidence",
    "spaciousness_mentions",
]

CATEGORICAL_FEATURES = ["area", "category"]

N_FOLDS = 5
MIN_RECOMMENDED_POSITIVES = 30

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def read_features():
    data = pd.read_csv(FEATURES_PATH)
    for col in NUMERIC_FEATURES + LABEL_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
    for col in CATEGORICAL_FEATURES + ["seat_type"]:
        data[col] = data[col].fillna("")
    return data


def expand_seat_type(data):
    seat_labels = data["seat_type"].apply(
        lambda value: [item for item in str(value).split(",") if item]
    )
    encoder = MultiLabelBinarizer()
    encoded = encoder.fit_transform(seat_labels)
    encoded_df = pd.DataFrame(
        encoded,
        columns=[f"seat_{label}" for label in encoder.classes_],
        index=data.index,
    )
    return pd.concat([data.drop(columns=["seat_type"]), encoded_df], axis=1), list(encoded_df.columns)


def make_pipeline(seat_columns):
    numeric_features = NUMERIC_FEATURES + seat_columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    classifier = OneVsRestClassifier(
        RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced_subsample",
        )
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", classifier),
        ]
    )


def multilabel_fold_indices(y_data, n_folds):
    """Greedy multilabel fold assignment that spreads sparse labels."""
    y_values = y_data.to_numpy(dtype=int)
    n_samples = len(y_values)
    folds = [[] for _ in range(n_folds)]
    fold_sizes = np.zeros(n_folds, dtype=int)
    fold_label_counts = np.zeros((n_folds, y_values.shape[1]), dtype=int)
    label_totals = y_values.sum(axis=0)

    sample_order = sorted(
        range(n_samples),
        key=lambda idx: (
            -sum(1 / max(label_totals[col], 1) for col in np.flatnonzero(y_values[idx])),
            idx,
        ),
    )

    for idx in sample_order:
        positive_cols = np.flatnonzero(y_values[idx])

        def fold_key(fold_idx):
            label_load = (
                fold_label_counts[fold_idx, positive_cols].sum()
                if len(positive_cols)
                else fold_label_counts[fold_idx].sum()
            )
            return (label_load, fold_sizes[fold_idx], fold_idx)

        target_fold = min(range(n_folds), key=fold_key)
        folds[target_fold].append(idx)
        fold_sizes[target_fold] += 1
        fold_label_counts[target_fold] += y_values[idx]

    all_indices = np.arange(n_samples)
    for val_idx in folds:
        val_idx = np.array(sorted(val_idx), dtype=int)
        train_idx = np.setdiff1d(all_indices, val_idx)
        yield train_idx, val_idx


def out_of_fold_predictions(x_data, y_data, seat_columns):
    """Generate out-of-fold predictions using multilabel-aware folds.

    Each restaurant's score comes from a model that did NOT see that
    restaurant during training, eliminating data leakage.
    """
    n_samples = len(x_data)
    n_labels = len(LABEL_COLUMNS)
    oof_scores = np.zeros((n_samples, n_labels))
    oof_preds = np.zeros((n_samples, n_labels), dtype=int)

    for fold_idx, (train_idx, val_idx) in enumerate(multilabel_fold_indices(y_data, N_FOLDS), start=1):
        logger.info("  fold %s/%s: train=%s, val=%s", fold_idx, N_FOLDS, len(train_idx), len(val_idx))

        fold_model = make_pipeline(seat_columns)
        fold_model.fit(x_data.iloc[train_idx], y_data.iloc[train_idx])

        oof_scores[val_idx] = fold_model.predict_proba(x_data.iloc[val_idx])
        oof_preds[val_idx] = fold_model.predict(x_data.iloc[val_idx])

    return oof_scores, oof_preds


def build_report(y_true, y_pred, total_size):
    per_label = {}
    warnings = []
    for i, label in enumerate(LABEL_COLUMNS):
        positive_count = int(y_true.iloc[:, i].sum())
        per_label[label] = {
            "positive_samples": positive_count,
            "needs_more_data": positive_count < MIN_RECOMMENDED_POSITIVES,
        }
        if positive_count < MIN_RECOMMENDED_POSITIVES:
            warnings.append(
                f"{label}: positive samples={positive_count}, recommended>={MIN_RECOMMENDED_POSITIVES}"
            )

    report = {
        "total_size": total_size,
        "n_folds": N_FOLDS,
        "method": "out-of-fold multilabel-aware cross validation (no data leakage)",
        "labels": LABEL_COLUMNS,
        "label_distribution": per_label,
        "warnings": warnings,
        "warning": "; ".join(warnings),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=LABEL_COLUMNS,
            zero_division=0,
            output_dict=True,
        ),
    }
    return report


def main():
    data = read_features()
    data, seat_columns = expand_seat_type(data)

    x_data = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES + seat_columns]
    y_data = data[LABEL_COLUMNS].astype(int)

    if len(data) < 10:
        raise ValueError("Need at least 10 restaurants to train a baseline model.")

    logger.info("dataset: %s restaurants", len(data))
    logger.info("label distribution:")
    for label in LABEL_COLUMNS:
        pos = int(y_data[label].sum())
        logger.info("  %s: %s positive (%.1f%%)", label, pos, pos / len(data) * 100)

    logger.info("\ngenerating out-of-fold predictions (%s-fold)...", N_FOLDS)
    oof_scores, oof_preds = out_of_fold_predictions(x_data, y_data, seat_columns)

    report = build_report(y_data, oof_preds, len(data))

    logger.info("\ntraining final model on all data...")
    model = make_pipeline(seat_columns)
    model.fit(x_data, y_data)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "label_columns": LABEL_COLUMNS,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "seat_columns": seat_columns,
        },
        MODEL_PATH,
    )

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    scores_df = pd.DataFrame(oof_scores, columns=[f"{label}_score" for label in LABEL_COLUMNS])
    output = pd.concat(
        [
            data[["restaurant_id", "restaurant_name", "area", "category", "rating", "review_count"]].reset_index(drop=True),
            scores_df,
        ],
        axis=1,
    )
    output.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")

    logger.info("\nsaved model: %s", MODEL_PATH)
    logger.info("saved report: %s", REPORT_PATH)
    logger.info("saved scores: %s", PREDICTIONS_PATH)

    logger.info("\n--- classification report (out-of-fold) ---")
    cr = report["classification_report"]
    for label in LABEL_COLUMNS:
        metrics = cr[label]
        dist = report["label_distribution"][label]
        logger.info(
            "  %-20s  P=%.2f  R=%.2f  F1=%.2f  (positive=%s)",
            label,
            metrics["precision"],
            metrics["recall"],
            metrics["f1-score"],
            dist["positive_samples"],
        )

    if report["warning"]:
        logger.warning("\nwarning: %s", report["warning"])


if __name__ == "__main__":
    main()
