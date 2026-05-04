import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, train_test_split
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


def out_of_fold_predictions(x_data, y_data, seat_columns):
    """Generate out-of-fold predictions using K-Fold cross validation.

    Each restaurant's score comes from a model that did NOT see that
    restaurant during training, eliminating data leakage.
    """
    n_samples = len(x_data)
    n_labels = len(LABEL_COLUMNS)
    oof_scores = np.zeros((n_samples, n_labels))
    oof_preds = np.zeros((n_samples, n_labels), dtype=int)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(x_data), start=1):
        print(f"  fold {fold_idx}/{N_FOLDS}: train={len(train_idx)}, val={len(val_idx)}")

        fold_model = make_pipeline(seat_columns)
        fold_model.fit(x_data.iloc[train_idx], y_data.iloc[train_idx])

        oof_scores[val_idx] = fold_model.predict_proba(x_data.iloc[val_idx])
        oof_preds[val_idx] = fold_model.predict(x_data.iloc[val_idx])

    return oof_scores, oof_preds


def build_report(y_true, y_pred, total_size):
    per_label = {}
    for i, label in enumerate(LABEL_COLUMNS):
        positive_count = int(y_true.iloc[:, i].sum())
        per_label[label] = {"positive_samples": positive_count}

    report = {
        "total_size": total_size,
        "n_folds": N_FOLDS,
        "method": "out-of-fold K-Fold cross validation (no data leakage)",
        "labels": LABEL_COLUMNS,
        "label_distribution": per_label,
        "warning": (
            "Dataset is very small, so these metrics are only a smoke test."
            if total_size < 100
            else ""
        ),
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

    print(f"dataset: {len(data)} restaurants")
    print(f"label distribution:")
    for label in LABEL_COLUMNS:
        pos = int(y_data[label].sum())
        print(f"  {label}: {pos} positive ({pos / len(data) * 100:.1f}%)")

    print(f"\ngenerating out-of-fold predictions ({N_FOLDS}-fold)...")
    oof_scores, oof_preds = out_of_fold_predictions(x_data, y_data, seat_columns)

    report = build_report(y_data, oof_preds, len(data))

    print("\ntraining final model on all data...")
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

    print(f"\nsaved model: {MODEL_PATH}")
    print(f"saved report: {REPORT_PATH}")
    print(f"saved scores: {PREDICTIONS_PATH}")

    print("\n--- classification report (out-of-fold) ---")
    cr = report["classification_report"]
    for label in LABEL_COLUMNS:
        metrics = cr[label]
        dist = report["label_distribution"][label]
        print(
            f"  {label:20s}  "
            f"P={metrics['precision']:.2f}  R={metrics['recall']:.2f}  "
            f"F1={metrics['f1-score']:.2f}  "
            f"(positive={dist['positive_samples']})"
        )

    if report["warning"]:
        print(f"\nwarning: {report['warning']}")


if __name__ == "__main__":
    main()
