import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYWORDS_PATH = ROOT / "config" / "keywords.json"
RAW_PATH = ROOT / "data" / "raw_reviews.csv"
SAMPLE_RAW_PATH = ROOT / "data" / "raw_reviews_sample.csv"
OUT_PATH = ROOT / "data" / "restaurants_features.csv"

SHRINKAGE_K = 5
LABEL_THRESHOLD = 0.05
NEGATION_WINDOW = 12


def load_keywords():
    with KEYWORDS_PATH.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def contains_any(text, words):
    text = str(text)
    return any(word in text for word in words)


def is_negated(text, keyword_start, negation_prefixes, keyword=""):
    """Check if a keyword is preceded by a nearby negation prefix."""
    start = max(0, keyword_start - NEGATION_WINDOW)
    prefix = text[start:keyword_start]
    suffix_start = keyword_start + len(keyword)
    suffix = text[suffix_start:suffix_start + NEGATION_WINDOW]
    return any(neg in prefix or neg in suffix for neg in negation_prefixes)


def keyword_positions(text, words):
    """Yield all keyword positions instead of only the first match."""
    text = str(text)
    for word in words:
        start = 0
        while word:
            idx = text.find(word, start)
            if idx < 0:
                break
            yield word, idx
            start = idx + len(word)


def seat_type_patterns(seat):
    return {
        "couple": [
            r"2\s*인(?:용)?\s*(?:석|자리|테이블)",
            r"두\s*명\s*(?:자리|테이블)",
        ],
        "group": [
            r"(?:단체|모임|회식)\s*(?:석|룸|자리|테이블)?",
            r"[4-9]\s*인(?:용)?\s*(?:석|자리|테이블)",
            r"\d+\s*명\s*(?:이상\s*)?(?:자리|테이블)",
        ],
        "bar": [r"(?:바|카운터)\s*(?:석|자리)"],
        "floor": [r"(?:좌식|바닥)\s*(?:석|자리)"],
        "room": [r"(?:룸|개별룸|프라이빗룸|방)\s*(?:석|자리)?"],
    }.get(seat, [])


def matches_seat_type(text, seat, words):
    if contains_any(text, words):
        return True
    return any(re.search(pattern, str(text)) for pattern in seat_type_patterns(seat))


def count_sentiment(text, positive_words, negative_words, negation_prefixes):
    """Count effective positive and negative signals in a single review.

    Returns (effective_positive, effective_negative) as booleans.
    - A positive keyword preceded by a negation counts as negative.
    - A negative keyword preceded by a negation counts as positive.
    """
    text = str(text)
    raw_pos = False
    negated_pos = False
    raw_neg = False
    negated_neg = False

    for word, idx in keyword_positions(text, positive_words):
        if is_negated(text, idx, negation_prefixes, word):
            negated_pos = True
        else:
            raw_pos = True

    for word, idx in keyword_positions(text, negative_words):
        if is_negated(text, idx, negation_prefixes, word):
            negated_neg = True
        else:
            raw_neg = True

    effective_pos = raw_pos or negated_neg
    effective_neg = raw_neg or negated_pos
    return effective_pos, effective_neg


def sentiment_score(reviews, positive_words, negative_words, negation_prefixes):
    if not reviews:
        return None, 0.0, 0
    pos = 0
    neg = 0
    for text in reviews:
        ep, en = count_sentiment(text, positive_words, negative_words, negation_prefixes)
        if ep:
            pos += 1
        if en:
            neg += 1
    mentions = pos + neg
    if mentions == 0:
        return None, 0.0, 0
    score = (pos + 1) / (mentions + 2)
    return score, mentions / len(reviews), mentions


def count_directional(text, high_words, low_words, negation_prefixes):
    """Count effective high and low signals in a single review.

    Same negation logic as count_sentiment:
    - A high keyword preceded by a negation counts as low.
    - A low keyword preceded by a negation counts as high.
    """
    text = str(text)
    raw_high = False
    negated_high = False
    raw_low = False
    negated_low = False

    for word, idx in keyword_positions(text, high_words):
        if is_negated(text, idx, negation_prefixes, word):
            negated_high = True
        else:
            raw_high = True

    for word, idx in keyword_positions(text, low_words):
        if is_negated(text, idx, negation_prefixes, word):
            negated_low = True
        else:
            raw_low = True

    effective_high = raw_high or negated_low
    effective_low = raw_low or negated_high
    return effective_high, effective_low


def directional_score(reviews, high_words, low_words, negation_prefixes):
    if not reviews:
        return None, 0.0, 0
    high = 0
    low = 0
    for text in reviews:
        eh, el = count_directional(text, high_words, low_words, negation_prefixes)
        if eh:
            high += 1
        if el:
            low += 1
    mentions = high + low
    if mentions == 0:
        return None, 0.0, 0
    score = (high + 1) / (mentions + 2)
    return score, mentions / len(reviews), mentions


def has_unnegated_label(text, label_words, negation_prefixes):
    for word, idx in keyword_positions(text, label_words):
        if not is_negated(str(text), idx, negation_prefixes, word):
            return True
    return False


def label_value(reviews, label_words, negation_prefixes=None):
    if not reviews:
        return 0
    negation_prefixes = negation_prefixes or []
    hits = sum(has_unnegated_label(text, label_words, negation_prefixes) for text in reviews)
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


def apply_bayesian_shrinkage(rows):
    """Apply Bayesian shrinkage to all score columns.

    Instead of a hard cutoff (mentions < 10 -> discard -> impute with mean),
    smoothly blend each restaurant's raw score with its group mean:

        adjusted = (mentions * raw_score + K * group_mean) / (mentions + K)

    Where K = SHRINKAGE_K (default 5). This means:
    - 0 mentions  -> pure group mean
    - 5 mentions  -> 50% own data, 50% group mean
    - 10 mentions -> 67% own data, 33% group mean
    - 30 mentions -> 86% own data, 14% group mean

    Compared to hard cutoff + mean imputation, this preserves partial
    information from low-mention restaurants instead of throwing it away.
    """
    if not rows:
        return

    score_cols = [col for col in rows[0] if col.endswith("_score")]
    mentions_map = {}
    for col in score_cols:
        base = col.replace("_score", "")
        mentions_col = f"{base}_mentions"
        if mentions_col in rows[0]:
            mentions_map[col] = mentions_col

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
            local = area_category_values[key][col]
            fallback = global_values[col]

            if local:
                group_mean = sum(local) / len(local)
            elif fallback:
                group_mean = sum(fallback) / len(fallback)
            else:
                group_mean = 0.5

            raw = to_float(row[col])
            mentions_col = mentions_map.get(col)
            mentions = int(to_float(row.get(mentions_col, 0)) or 0) if mentions_col else 0

            if raw is None:
                row[col] = group_mean
            else:
                row[col] = (mentions * raw + SHRINKAGE_K * group_mean) / (mentions + SHRINKAGE_K)


def build_features(raw_rows, keywords):
    rows = []
    grouped = group_by_restaurant(raw_rows)
    negation_prefixes = keywords.get("negation_prefixes", [])

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
                reviews,
                features[name]["positive"],
                features[name]["negative"],
                negation_prefixes,
            )
            row[f"{name}_score"] = score
            row[f"{name}_confidence"] = confidence
            row[f"{name}_mentions"] = mentions

        brightness, confidence, mentions = directional_score(
            reviews,
            features["brightness"]["bright"],
            features["brightness"]["dark"],
            negation_prefixes,
        )
        row["brightness_score"] = brightness
        row["brightness_confidence"] = confidence
        row["brightness_mentions"] = mentions

        noise, confidence, mentions = directional_score(
            reviews,
            features["noise"]["lively"],
            features["noise"]["quiet"],
            negation_prefixes,
        )
        row["noise_score"] = noise
        row["noise_confidence"] = confidence
        row["noise_mentions"] = mentions

        spaciousness, confidence, mentions = directional_score(
            reviews,
            features["spaciousness"]["spacious"],
            features["spaciousness"]["cozy"],
            negation_prefixes,
        )
        row["spaciousness_score"] = spaciousness
        row["spaciousness_confidence"] = confidence
        row["spaciousness_mentions"] = mentions

        seat_hits = []
        for seat, words in features["seat_type"].items():
            if any(matches_seat_type(text, seat, words) for text in reviews):
                seat_hits.append(seat)
        row["seat_type"] = ",".join(seat_hits)

        for label, words in keywords["labels"].items():
            row[label] = label_value(reviews, words, negation_prefixes)

        rows.append(row)

    apply_bayesian_shrinkage(rows)
    return rows


def write_features(rows):
    if not rows:
        raise ValueError("No feature rows to write. Check raw review input.")
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
