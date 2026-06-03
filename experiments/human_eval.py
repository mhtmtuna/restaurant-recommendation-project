"""B6. Human Evaluation 템플릿 생성

18개 조건 조합 (3 지역 × 6 상황) × top-5 추천 결과를 CSV로 출력.
팀원이 각 추천 결과에 1~5점 평가 후 human_eval_filled.csv로 저장.
결과: experiments/human_eval_template.csv
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SCORES_PATH = ROOT / "data" / "restaurant_label_scores.csv"
OUT_DIR = ROOT / "experiments"
OUT_TEMPLATE = OUT_DIR / "human_eval_template.csv"
OUT_SUMMARY = OUT_DIR / "human_eval_summary.csv"

TOP_K = 5

CONDITIONS = [
    ("강남", "연인",     "식사/술자리", "couple_score"),
    ("강남", "친구",     "식사",        "friend_meal_score"),
    ("강남", "친구",     "술자리",      "friend_drink_score"),
    ("강남", "비즈니스", "식사",        "business_meal_score"),
    ("강남", "비즈니스", "술자리",      "business_drink_score"),
    ("건대", "연인",     "식사/술자리", "couple_score"),
    ("건대", "친구",     "식사",        "friend_meal_score"),
    ("건대", "친구",     "술자리",      "friend_drink_score"),
    ("건대", "비즈니스", "식사",        "business_meal_score"),
    ("건대", "비즈니스", "술자리",      "business_drink_score"),
    ("잠실", "연인",     "식사/술자리", "couple_score"),
    ("잠실", "친구",     "식사",        "friend_meal_score"),
    ("잠실", "친구",     "술자리",      "friend_drink_score"),
    ("잠실", "비즈니스", "식사",        "business_meal_score"),
    ("잠실", "비즈니스", "술자리",      "business_drink_score"),
]


def get_top_k(scores_df, area, score_col, k=TOP_K):
    mask = scores_df["area"].str.contains(area, na=False)
    filtered = scores_df[mask].sort_values(score_col, ascending=False).head(k)
    return filtered.reset_index(drop=True)


def build_template(scores_df, k=TOP_K):
    rows = []
    for area, relation, occasion, score_col in CONDITIONS:
        top = get_top_k(scores_df, area, score_col, k)
        for rank, r in enumerate(top.itertuples(index=False), start=1):
            rows.append({
                "조건_지역": area,
                "조건_관계": relation,
                "조건_목적": occasion,
                "순위": rank,
                "식당명": r.restaurant_name,
                "카테고리": r.category,
                "추천점수": round(getattr(r, score_col), 4),
                "별점": r.rating,
                "리뷰수": r.review_count,
                # 팀원이 채울 열
                "적절성_1~5": "",
                "평가자": "",
                "메모": "",
            })
    return pd.DataFrame(rows)


def summarize(filled_path: Path):
    """평가가 완료된 CSV를 집계해 요약 출력."""
    df = pd.read_csv(filled_path, encoding="utf-8-sig")
    df["적절성_1~5"] = pd.to_numeric(df["적절성_1~5"], errors="coerce")
    summary = (
        df.groupby(["조건_관계", "조건_목적", "조건_지역"])["적절성_1~5"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "평균점수", "count": "평가수"})
        .reset_index()
    )
    summary["평균점수"] = summary["평균점수"].round(2)
    print(summary.to_string(index=False))
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
    print(f"\nsaved: {OUT_SUMMARY}")


def main():
    import sys
    args = sys.argv[1:]
    if args and args[0] == "--summarize":
        filled = Path(args[1]) if len(args) > 1 and not args[1].startswith("--") else OUT_DIR / "human_eval_filled.csv"
        summarize(filled)
        return

    # 옵션: --topk N (기본 5), --out 파일명 (기본 human_eval_template.csv)
    top_k = TOP_K
    out_path = OUT_TEMPLATE
    if "--topk" in args:
        top_k = int(args[args.index("--topk") + 1])
    if "--out" in args:
        out_arg = Path(args[args.index("--out") + 1])
        out_path = out_arg if out_arg.is_absolute() else OUT_DIR / out_arg

    scores_df = pd.read_csv(SCORES_PATH)
    template = build_template(scores_df, top_k)
    template.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"saved: {out_path}")
    print(f"\n총 {len(template)}행 ({len(CONDITIONS)}개 조건 × top-{top_k})")
    print("\n[사용법]")
    print(f"  1. {out_path.name} 열기")
    print("  2. '적절성_1~5' 열에 1(매우 부적절) ~ 5(매우 적절) 입력")
    print("  3. '평가자' 열에 이름 입력")
    print(f"  4. experiments/human_eval_filled.csv 로 저장 후 아래 실행:")
    print("     python experiments/human_eval.py --summarize")

    print("\n=== 미리보기 (조건별 top-1) ===")
    preview = template[template["순위"] == 1][
        ["조건_지역", "조건_관계", "조건_목적", "식당명", "카테고리", "추천점수"]
    ]
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
