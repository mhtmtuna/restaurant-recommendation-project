"""
여러 raw_reviews_*.csv 파일을 하나의 raw_reviews.csv로 병합.

사용법:
    python src/merge_reviews.py
"""

import csv
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_PATH = DATA_DIR / "raw_reviews.csv"
DEFAULT_SOURCE_PATHS = [
    DATA_DIR / "raw_reviews_kakao.csv",
    DATA_DIR / "raw_reviews_naver.csv",
]

FIELDNAMES = [
    "restaurant_id", "restaurant_name", "area", "category",
    "rating", "review_count", "price", "photo_ratio", "review_text",
]


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def dedupe(rows):
    seen, result = set(), []
    for row in rows:
        key = (row.get("restaurant_id", ""), row.get("review_text", ""))
        if key[0] and key[1] and key not in seen:
            seen.add(key)
            result.append(row)
    return result


def resolve_sources(explicit_sources=None):
    if explicit_sources:
        return [Path(source) for source in explicit_sources]

    sources = [path for path in DEFAULT_SOURCE_PATHS if path.exists()]
    if (DATA_DIR / "raw_reviews_kakao.csv").exists():
        return sources

    if OUTPUT_PATH.exists():
        print(
            "[warn] data/raw_reviews_kakao.csv가 없어 기존 data/raw_reviews.csv를 "
            "카카오/기존 소스 fallback으로 사용합니다. 원천 파일과 병합 산출물을 "
            "분리하려면 data/raw_reviews_kakao.csv를 두거나 --source를 지정하세요."
        )
        return [OUTPUT_PATH] + [path for path in sources if path != OUTPUT_PATH]

    return sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        help="병합할 원천 CSV 경로. 여러 번 지정 가능.",
    )
    args = parser.parse_args()

    sources = [path for path in resolve_sources(args.source) if path.exists()]
    if not sources:
        print("병합할 파일이 없습니다.")
        return

    print("병합 대상 파일:")
    all_rows = []
    for path in sources:
        rows = read_csv(path)
        print(f"  {path.name}: {len(rows)}행")
        all_rows.extend(rows)

    merged = dedupe(all_rows)
    print(f"\n중복 제거 후: {len(merged)}행")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)

    print(f"저장 완료: {OUTPUT_PATH}")

    # 지역·카테고리별 분포
    from collections import Counter
    areas = Counter(r["area"] for r in merged)
    cats = Counter(r["category"] for r in merged)
    rests = len({r["restaurant_id"] for r in merged})

    print(f"\n식당 수: {rests}개  |  리뷰 수: {len(merged)}개")
    print("\n지역별:")
    for area, cnt in sorted(areas.items(), key=lambda x: -x[1]):
        print(f"  {area}: {cnt}")
    print("\n카테고리별:")
    for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt}")


if __name__ == "__main__":
    main()
