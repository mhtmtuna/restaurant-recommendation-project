"""베이스라인 재계산 — 최종 5라벨 + model_report와 동일한 sklearn macro F1.

baseline_comparison.py(구버전, 6라벨, import 깨짐)를 대체.
'ours'(RF)는 재학습하지 않고 data/model_report.json 의 확정 수치(0.619)를 사용한다.
출력: experiments/baseline_recompute.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_training_common import LABEL_COLUMNS, read_features, expand_seat_type  # noqa: E402


def mf1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def main():
    data = read_features()
    data, _ = expand_seat_type(data)
    y = data[LABEL_COLUMNS].astype(int).to_numpy()
    n, k = y.shape

    # Random: 라벨별 양성 비율로 베르누이 샘플
    rng = np.random.default_rng(42)
    rates = y.mean(axis=0)
    rand = np.stack([rng.binomial(1, r, size=n) for r in rates], axis=1)

    # Majority: 전부 0
    major = np.zeros_like(y)

    # Rating-only: 별점>=4.0 이면 전 라벨 1 (결측/오염 rating은 sanitize로 NaN -> False)
    rating = data["rating"].to_numpy()
    high = np.nan_to_num(rating, nan=-1.0) >= 4.0
    rate_only = np.stack([high.astype(int)] * k, axis=1)

    # Keyword-only: 라벨이 키워드 산출물이라 정답=예측 -> 1.0 (순환 증거)
    kw = y.copy()

    results = {
        "Random": mf1(y, rand),
        "Majority (all-zero)": mf1(y, major),
        "Rating-only (>=4.0)": mf1(y, rate_only),
        "Keyword-only": mf1(y, kw),
    }

    # ours: 확정 리포트에서
    rep = json.load(open(ROOT / "data" / "model_report.json", encoding="utf-8"))
    ours = rep["evaluation"]["macro_f1"]
    results["RF (ours, from model_report)"] = ours

    print(f"labels({k}): {LABEL_COLUMNS}")
    print(f"samples: {n}\n")
    for name, v in results.items():
        print(f"  {name:<32} macro_f1 = {v:.4f}")

    ratio = ours / results["Rating-only (>=4.0)"] if results["Rating-only (>=4.0)"] else float("inf")
    print(f"\n  RF / Rating-only = {ratio:.2f}x")
    print(f"  RF / Random      = {ours / results['Random']:.2f}x" if results["Random"] else "")

    out = {"macro_f1": {k_: round(v, 4) for k_, v in results.items()},
           "ratios": {"RF_vs_RatingOnly": round(ratio, 2),
                      "RF_vs_Random": round(ours / results["Random"], 2) if results["Random"] else None}}
    (ROOT / "experiments" / "baseline_recompute.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: experiments/baseline_recompute.json")


if __name__ == "__main__":
    main()
