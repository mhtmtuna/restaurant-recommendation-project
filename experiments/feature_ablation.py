"""Evaluate Random Forest feature-removal candidates with OOF predictions."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, r2_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_training_common import (
    CATEGORICAL_FEATURES,
    LABEL_COLUMNS,
    NUMERIC_FEATURES,
    N_FOLDS,
    expand_seat_type,
    multilabel_fold_indices,
    read_features,
)
from train_model import estimator_factory

OUT_CSV = ROOT / "experiments" / "feature_ablation.csv"
OUT_JSON = ROOT / "experiments" / "feature_ablation.json"

EXPERIMENTS = {
    "baseline": [],
    "drop_taste_score": ["taste_score"],
    "drop_value_confidence": ["value_confidence"],
    "drop_brightness_confidence": ["brightness_confidence"],
    "drop_portion_mentions": ["portion_mentions"],
    "drop_brightness_mentions": ["brightness_mentions"],
    "drop_non_significant_group": [
        "taste_score",
        "value_confidence",
        "brightness_confidence",
        "portion_mentions",
        "brightness_mentions",
    ],
    "drop_collected_review_count": ["collected_review_count"],
    "drop_review_count": ["review_count"],
    "drop_rating": ["rating"],
    "drop_collection_metadata": ["rating", "review_count", "collected_review_count"],
}


def make_pipeline(numeric_features, seat_columns):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features + seat_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0.0,
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", OneVsRestClassifier(estimator_factory())),
        ]
    )


def evaluate(data, y_data, seat_columns, dropped):
    numeric_features = [feature for feature in NUMERIC_FEATURES if feature not in dropped]
    x_data = data[numeric_features + CATEGORICAL_FEATURES + seat_columns]
    scores = np.zeros((len(data), len(LABEL_COLUMNS)))
    preds = np.zeros((len(data), len(LABEL_COLUMNS)), dtype=int)

    for train_idx, val_idx in multilabel_fold_indices(y_data, N_FOLDS):
        model = make_pipeline(numeric_features, seat_columns)
        model.fit(x_data.iloc[train_idx], y_data.iloc[train_idx])
        scores[val_idx] = model.predict_proba(x_data.iloc[val_idx])
        preds[val_idx] = model.predict(x_data.iloc[val_idx])

    return {
        "r2_score": float(r2_score(y_data, scores, multioutput="uniform_average")),
        "macro_f1": float(f1_score(y_data, preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_data, preds, average="micro", zero_division=0)),
    }


def main():
    data = read_features()
    data, seat_columns = expand_seat_type(data)
    y_data = data[LABEL_COLUMNS].astype(int)

    rows = []
    for name, dropped in EXPERIMENTS.items():
        print(f"evaluating: {name}")
        metrics = evaluate(data, y_data, seat_columns, dropped)
        rows.append({"experiment": name, "dropped_features": ", ".join(dropped), **metrics})

    result = pd.DataFrame(rows)
    baseline = result.iloc[0]
    result["delta_r2"] = result["r2_score"] - baseline["r2_score"]
    result["delta_macro_f1"] = result["macro_f1"] - baseline["macro_f1"]
    result.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(
        json.dumps(result.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"saved: {OUT_CSV}")
    print(f"saved: {OUT_JSON}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
