import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
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


def probability_frame(model, x_data):
    probabilities = model.predict_proba(x_data)
    return pd.DataFrame(probabilities, columns=[f"{label}_score" for label in LABEL_COLUMNS])


def build_report(y_true, y_pred, train_size, test_size):
    return {
        "train_size": train_size,
        "test_size": test_size,
        "labels": LABEL_COLUMNS,
        "warning": (
            "Dataset is very small, so these metrics are only a smoke test."
            if train_size + test_size < 100
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


def main():
    data = read_features()
    data, seat_columns = expand_seat_type(data)

    x_data = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES + seat_columns]
    y_data = data[LABEL_COLUMNS].astype(int)

    if len(data) < 10:
        raise ValueError("Need at least 10 restaurants to train a baseline model.")

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=0.25,
        random_state=42,
    )

    eval_model = make_pipeline(seat_columns)
    eval_model.fit(x_train, y_train)

    y_pred = eval_model.predict(x_test)
    report = build_report(y_test, y_pred, len(x_train), len(x_test))

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

    scores = probability_frame(model, x_data)
    output = pd.concat(
        [
            data[["restaurant_id", "restaurant_name", "area", "category", "rating", "review_count"]],
            scores,
        ],
        axis=1,
    )
    output.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")

    print(f"saved model: {MODEL_PATH}")
    print(f"saved report: {REPORT_PATH}")
    print(f"saved scores: {PREDICTIONS_PATH}")
    if report["warning"]:
        print(f"warning: {report['warning']}")


if __name__ == "__main__":
    main()
