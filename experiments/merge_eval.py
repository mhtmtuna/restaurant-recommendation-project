"""팀원들이 제출한 human_eval CSV를 합쳐서 summarize 실행.

사용법:
    python experiments/merge_eval.py 팀원A.csv 팀원B.csv 팀원C.csv
    python experiments/merge_eval.py  # experiments/ 폴더의 *_filled.csv 전부 자동 수집
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
OUT_MERGED = EXPERIMENTS / "human_eval_filled.csv"
OUT_SUMMARY = EXPERIMENTS / "human_eval_summary.csv"


def merge(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p, encoding="utf-8-sig")
        if "적절성_1~5" not in df.columns:
            print(f"[경고] {p.name}: '적절성_1~5' 열 없음 — 건너뜀")
            continue
        filled = df[df["적절성_1~5"].notna() & (df["적절성_1~5"] != "")]
        print(f"  {p.name}: {len(filled)}행 수집 (전체 {len(df)}행)")
        frames.append(df)
    if not frames:
        raise SystemExit("수집된 데이터가 없습니다.")
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame):
    df["적절성_1~5"] = pd.to_numeric(df["적절성_1~5"], errors="coerce")
    summary = (
        df.groupby(["조건_관계", "조건_목적", "조건_지역"])["적절성_1~5"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "평균점수", "count": "평가수"})
        .reset_index()
    )
    summary["평균점수"] = summary["평균점수"].round(2)
    print("\n=== Human Evaluation 결과 ===")
    print(summary.to_string(index=False))
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
    print(f"\nsaved: {OUT_SUMMARY}")


def main():
    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        paths = sorted(EXPERIMENTS.glob("*_filled.csv"))
        if not paths:
            raise SystemExit(
                "파일을 찾을 수 없습니다.\n"
                "사용법: python experiments/merge_eval.py 팀원A.csv 팀원B.csv ..."
            )

    print(f"수집할 파일 {len(paths)}개:")
    merged = merge(paths)
    merged.to_csv(OUT_MERGED, index=False, encoding="utf-8-sig")
    print(f"\n합친 파일 저장: {OUT_MERGED} ({len(merged)}행)")
    summarize(merged)


if __name__ == "__main__":
    main()
