"""Evaluate whether the Random Forest generalizes across review sources."""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, f1_score, r2_score

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

REPORT_PATH = ROOT / "data" / "model_report.json"
OUT_JSON = ROOT / "experiments" / "source_aware_evaluation.json"
OUT_CSV = ROOT / "experiments" / "source_aware_evaluation.csv"


def source_mask(data, source):
    is_naver = data["restaurant_id"].astype(str).str.startswith("nv_")
    return is_naver if source == "naver" else ~is_naver


def positive_rates(y_data):
    return {label: float(y_data[label].mean()) for label in LABEL_COLUMNS}


def evaluate_direction(data, seat_columns, train_source, test_source):
    train = data[source_mask(data, train_source)]
    test = data[source_mask(data, test_source)]
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + seat_columns
    x_train = train[feature_columns]
    x_test = test[feature_columns]
    y_train = train[LABEL_COLUMNS].astype(int)
    y_test = test[LABEL_COLUMNS].astype(int)

    model = make_pipeline(seat_columns, estimator_factory())
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_test)
    preds = model.predict(x_test)
    report = classification_report(
        y_test,
        preds,
        target_names=LABEL_COLUMNS,
        zero_division=0,
        output_dict=True,
    )

    return {
        "evaluation": f"train_{train_source}_test_{test_source}",
        "train_source": train_source,
        "test_source": test_source,
        "train_size": len(train),
        "test_size": len(test),
        "train_positive_rates": positive_rates(y_train),
        "test_positive_rates": positive_rates(y_test),
        "r2_score": float(r2_score(y_test, scores, multioutput="uniform_average")),
        "macro_f1": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_test, preds, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(y_test, preds, average="samples", zero_division=0)),
        "per_label": {
            label: {
                "r2_score": float(r2_score(y_test[label], scores[:, index])),
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1_score": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for index, label in enumerate(LABEL_COLUMNS)
        },
    }


def mixed_oof_reference():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    evaluation = report["evaluation"]
    return {
        "evaluation": "mixed_source_oof_reference",
        "train_source": "mixed",
        "test_source": "mixed",
        "train_size": report["total_size"],
        "test_size": report["total_size"],
        "r2_score": evaluation["r2_score"],
        "macro_f1": evaluation["macro_f1"],
        "micro_f1": evaluation["micro_f1"],
        "samples_f1": evaluation["samples_f1"],
    }


def main():
    data = read_features()
    data, seat_columns = expand_seat_type(data)
    results = [
        mixed_oof_reference(),
        evaluate_direction(data, seat_columns, "kakao", "naver"),
        evaluate_direction(data, seat_columns, "naver", "kakao"),
    ]

    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = pd.DataFrame(results)[
        [
            "evaluation",
            "train_source",
            "test_source",
            "train_size",
            "test_size",
            "r2_score",
            "macro_f1",
            "micro_f1",
            "samples_f1",
        ]
    ]
    summary.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"saved: {OUT_JSON}")
    print(f"saved: {OUT_CSV}")
    print("\n=== source-aware evaluation ===")
    print(summary.to_string(index=False))

    for result in results[1:]:
        print(f"\n=== {result['evaluation']} per label ===")
        print(pd.DataFrame(result["per_label"]).T.to_string())


if __name__ == "__main__":
    main()
