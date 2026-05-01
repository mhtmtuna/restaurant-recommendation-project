import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYWORDS_PATH = ROOT / "config" / "keywords.json"
RAW_PATH = ROOT / "data" / "raw_reviews.csv"
SAMPLE_RAW_PATH = ROOT / "data" / "raw_reviews_sample.csv"
OUT_PATH = ROOT / "data" / "restaurants_features.csv"

MIN_MENTIONS = 10
LABEL_THRESHOLD = 0.05


def load_keywords():
    with KEYWORDS_PATH.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def contains_any(text, words):
    text = str(text)
    return any(word in text for word in words)


def sentiment_score(reviews, positive_words, negative_words):
    pos = sum(contains_any(text, positive_words) for text in reviews)
    neg = sum(contains_any(text, negative_words) for text in reviews)
    mentions = pos + neg
    if mentions == 0:
        return None, 0.0, 0
    return pos / mentions, mentions / len(reviews), mentions


def directional_score(reviews, high_words, low_words):
    high = sum(contains_any(text, high_words) for text in reviews)
    low = sum(contains_any(text, low_words) for text in reviews)
    mentions = high + low
    if mentions == 0:
        return None, 0.0, 0
    return high / mentions, mentions / len(reviews), mentions


def label_value(reviews, label_words):
    hits = sum(contains_any(text, label_words) for text in reviews)
    return int((hits / len(reviews)) >= LABEL_THRESHOLD)


def read_raw_reviews():
    path = RAW_PATH if RAW_PATH.exists() else SAMPLE_RAW_PATH
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def group_by_restaurant(raw_rows):
    grouped = defaultdict(list)
    for row in raw_rows:
        grouped[row["restaurant_id"]].append(row)
    return grouped


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fill_missing_scores(rows):
    score_cols = [col for col in rows[0] if col.endswith("_score")]
    area_category_values = defaultdict(lambda: defaultdict(list))

    for row in rows:
        key = (row["area"], row["category"])
        for col in score_cols:
            value = to_float(row[col])
            if value is not None:
                area_category_values[key][col].append(value)

    global_values = defaultdict(list)
    for row in rows:
        for col in score_cols:
            value = to_float(row[col])
            if value is not None:
                global_values[col].append(value)

    for row in rows:
        key = (row["area"], row["category"])
        for col in score_cols:
            if row[col] is not None:
                continue
            local = area_category_values[key][col]
            fallback = global_values[col]
            if local:
                row[col] = sum(local) / len(local)
            elif fallback:
                row[col] = sum(fallback) / len(fallback)
            else:
                row[col] = ""


def build_features(raw_rows, keywords):
    rows = []
    grouped = group_by_restaurant(raw_rows)

    for restaurant_id, group in grouped.items():
        reviews = [row["review_text"] for row in group if row.get("review_text")]
        first = group[0]
        row = {
            "restaurant_id": restaurant_id,
            "restaurant_name": first["restaurant_name"],
            "area": first["area"],
            "category": first["category"],
            "price": first["price"],
            "rating": first["rating"],
            "review_count": first["review_count"],
            "photo_ratio": first["photo_ratio"],
            "collected_review_count": len(reviews),
        }

        features = keywords["features"]
        for name in ["taste", "value", "portion"]:
            score, confidence, mentions = sentiment_score(
                reviews, features[name]["positive"], features[name]["negative"]
            )
            row[f"{name}_score"] = score if mentions >= MIN_MENTIONS else None
            row[f"{name}_confidence"] = confidence

        brightness, confidence, mentions = directional_score(
            reviews, features["brightness"]["bright"], features["brightness"]["dark"]
        )
        row["brightness_score"] = brightness if mentions >= MIN_MENTIONS else None
        row["brightness_confidence"] = confidence

        noise, confidence, mentions = directional_score(
            reviews, features["noise"]["lively"], features["noise"]["quiet"]
        )
        row["noise_score"] = noise if mentions >= MIN_MENTIONS else None
        row["noise_confidence"] = confidence

        spaciousness, confidence, mentions = directional_score(
            reviews, features["spaciousness"]["spacious"], features["spaciousness"]["cozy"]
        )
        row["spaciousness_score"] = spaciousness if mentions >= MIN_MENTIONS else None
        row["spaciousness_confidence"] = confidence

        seat_hits = []
        for seat, words in features["seat_type"].items():
            if any(contains_any(text, words) for text in reviews):
                seat_hits.append(seat)
        row["seat_type"] = ",".join(seat_hits)

        for label, words in keywords["labels"].items():
            row[label] = label_value(reviews, words)

        rows.append(row)

    fill_missing_scores(rows)
    return rows


def write_features(rows):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with OUT_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    keywords = load_keywords()
    raw_rows = read_raw_reviews()
    features = build_features(raw_rows, keywords)
    write_features(features)
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
