import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCORES_PATH = ROOT / "data" / "restaurant_label_scores.csv"

LABEL_MAP = {
    ("연인", "식사"): "couple_meal_score",
    ("연인", "술자리"): "couple_drink_score",
    ("친구", "식사"): "friend_meal_score",
    ("친구", "술자리"): "friend_drink_score",
    ("비즈니스", "식사"): "business_meal_score",
    ("비즈니스", "술자리"): "business_drink_score",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relation", required=True, choices=["연인", "친구", "비즈니스"])
    parser.add_argument("--occasion", required=True, choices=["식사", "술자리"])
    parser.add_argument("--area", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    score_col = LABEL_MAP[(args.relation, args.occasion)]
    scores = pd.read_csv(SCORES_PATH)

    query = args.area.replace("구", "").replace("동", "")
    mask = scores["area"].str.contains(query, na=False) | scores["area"].str.contains(args.area, na=False)
    filtered = scores[mask].copy()

    if filtered.empty:
        available = sorted(scores["area"].dropna().unique())
        print(f"'{args.area}'에 해당하는 식당이 없습니다.")
        print(f"사용 가능한 지역: {', '.join(available)}")
        return

    filtered = filtered.sort_values(score_col, ascending=False).head(args.top_k)

    for idx, row in enumerate(filtered.itertuples(index=False), start=1):
        print(
            f"{idx}. {row.restaurant_name} "
            f"({row.area}, {row.category}) - score {getattr(row, score_col):.3f}"
        )


if __name__ == "__main__":
    main()
