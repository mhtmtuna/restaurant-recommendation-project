"""
망고플레이트 리뷰 크롤러
- 출력 포맷: raw_reviews.csv 와 동일 (restaurant_id, restaurant_name, area, category,
  rating, review_count, price, photo_ratio, review_text)
- 수집 결과: data/raw_reviews_mangoplate.csv
- 진행 상태: data/crawl_status_mangoplate.csv  (중단 후 재시작 가능)
- 에러 로그: data/crawl_errors_mangoplate.csv
"""

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

from selenium import webdriver
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    JavascriptException,
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "config" / "sampling_plan.json"
OUTPUT_PATH = ROOT / "data" / "raw_reviews_mangoplate.csv"
STATUS_PATH = ROOT / "data" / "crawl_status_mangoplate.csv"
ERRORS_PATH = ROOT / "data" / "crawl_errors_mangoplate.csv"
BASE_URL = "https://www.mangoplate.com"

FIELDNAMES = [
    "restaurant_id", "restaurant_name", "area", "category",
    "rating", "review_count", "price", "photo_ratio", "review_text",
]
STATUS_FIELDNAMES = [
    "area", "category", "restaurant_id", "restaurant_name",
    "status", "collected_count", "target_count",
]
ERROR_FIELDNAMES = ["area", "category", "restaurant_id", "restaurant_name", "stage", "error"]

# 망고플레이트 카카오 카테고리 → 검색 키워드 매핑
CATEGORY_SEARCH = {
    "한식": "한식",
    "일식": "일식",
    "중식": "중식",
    "양식": "양식",
    "술집": "술집",
    "치킨": "치킨",
    "분식": "분식",
    "이자카야": "이자카야",
    "고깃집": "고기",
    "호프/통닭": "호프",
    "카페 디저트": "카페",
}


@dataclass
class Place:
    place_id: str
    name: str
    url: str
    rating: str = ""
    review_count: str = ""


# ── 드라이버 ──────────────────────────────────────────────────────────────────

def make_driver(headless=True):
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--lang=ko-KR")
    options.add_argument("--window-size=1400,1000")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(options=options)


def safe_click(driver, element):
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
    time.sleep(0.3)
    try:
        element.click()
    except (ElementClickInterceptedException, StaleElementReferenceException):
        driver.execute_script("arguments[0].click();", element)


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def digits(text):
    m = re.search(r"[\d,]+", text or "")
    return m.group(0).replace(",", "") if m else ""


def extract_place_id(url):
    m = re.search(r"/restaurants/([^/?#]+)", url)
    return f"mp_{m.group(1)}" if m else f"mp_{re.sub(r'[^a-zA-Z0-9]', '_', url)[-30:]}"


def text_first(elements, selectors):
    for sel in selectors:
        els = elements.find_elements(By.CSS_SELECTOR, sel)
        if els:
            t = els[0].text.strip()
            if t:
                return t
    return ""


# ── 검색 ──────────────────────────────────────────────────────────────────────

def search_places(driver, area, category, limit):
    search_kw = CATEGORY_SEARCH.get(category, category)
    query = f"{area} {search_kw}"
    url = f"{BASE_URL}/search?keyword={quote_plus(query)}"
    driver.get(url)

    wait = WebDriverWait(driver, 15)
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,
            ".ResItemBox, .list-restaurant-item, .item-restaurant, article")))
    except TimeoutException:
        print(f"  [warn] 검색 결과 없음: {query}")
        return []

    places = []
    seen = set()
    stuck = 0

    for page in range(1, 15):
        card_selectors = (
            ".ResItemBox, .list-restaurant-item, "
            "article[class*='restaurant'], li[class*='restaurant'], "
            ".restaurant-item"
        )
        cards = driver.find_elements(By.CSS_SELECTOR, card_selectors)
        prev_len = len(places)

        for card in cards:
            name = text_first(card, [
                ".RestaurantName", ".restaurant-name", "h2", "h3",
                "[class*='title']", "[class*='name']",
            ])
            link = ""
            for a in card.find_elements(By.CSS_SELECTOR, "a"):
                href = a.get_attribute("href") or ""
                if "/restaurants/" in href:
                    link = href
                    break
            if not name or not link:
                continue
            place_id = extract_place_id(link)
            if place_id in seen:
                continue
            seen.add(place_id)

            rating = text_first(card, [
                ".RatingNumber", "[class*='rating']", "[class*='score']", ".rate",
            ])
            review_count = digits(text_first(card, [
                "[class*='review']", "[class*='comment']", "[class*='count']",
            ]))
            places.append(Place(place_id=place_id, name=name, url=link,
                                rating=rating, review_count=review_count))
            if len(places) >= limit:
                return places

        if len(places) == prev_len:
            stuck += 1
            if stuck >= 2:
                break
        else:
            stuck = 0

        # 다음 페이지
        next_url = f"{BASE_URL}/search?keyword={quote_plus(query)}&page={page + 1}"
        driver.get(next_url)
        time.sleep(1.5)
        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,
                ".ResItemBox, .list-restaurant-item, article")))
        except TimeoutException:
            break

    return places


# ── 식당 페이지: 메타 정보 ────────────────────────────────────────────────────

def extract_meta(driver):
    rating = text_first(driver, [
        ".RatingNumber", "[class*='RatingAvg']", "[class*='rating-number']",
        "[class*='score-avg']", ".avg-rating",
    ])
    rating = re.sub(r"[^\d.]", "", rating)

    review_count_text = text_first(driver, [
        "[class*='ReviewCount']", "[class*='review-count']",
        "[class*='total-review']", "[class*='reviewCnt']",
    ])
    review_count = digits(review_count_text)

    price = ""
    for sel in ["[class*='price']", "[class*='Price']", ".menu-price", "[class*='cost']"]:
        for el in driver.find_elements(By.CSS_SELECTOR, sel):
            t = el.text.strip()
            m = re.search(r"[\d,]{4,}", t)
            if m:
                val = int(m.group(0).replace(",", ""))
                if 1000 <= val <= 500000:
                    price = str(val)
                    break
        if price:
            break

    return rating, review_count, price


# ── 리뷰 수집 ─────────────────────────────────────────────────────────────────

def _collect_visible_reviews(driver):
    selectors = [
        ".Review .review_contents",
        ".Review p",
        "[class*='ReviewItem'] p",
        "[class*='ReviewItem'] [class*='content']",
        "[class*='review-item'] p",
        "[class*='review-item'] [class*='content']",
        "[class*='review'] [class*='text']",
        ".RestaurantReviewItem p",
    ]
    texts = []
    seen = set()
    for sel in selectors:
        for el in driver.find_elements(By.CSS_SELECTOR, sel):
            t = el.text.strip()
            if len(t) >= 5 and t not in seen:
                seen.add(t)
                texts.append(t)
    return texts


def collect_reviews(driver, url, target_count):
    driver.get(url)
    wait = WebDriverWait(driver, 12)

    # 리뷰 탭 또는 리뷰 섹션이 로드될 때까지 대기
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,
            ".Review, [class*='ReviewItem'], [class*='review-item'], [class*='ReviewList']")))
    except TimeoutException:
        time.sleep(2.0)

    # 리뷰 탭 클릭
    for sel in ["a[href*='review']", "button[class*='review']", "[data-tab='review']",
                "li[class*='review'] a", ".tab-review"]:
        els = driver.find_elements(By.CSS_SELECTOR, sel)
        for el in els:
            if el.is_displayed():
                try:
                    safe_click(driver, el)
                    time.sleep(1.0)
                    break
                except Exception:
                    pass

    seen = set()
    stuck = 0

    for _ in range(20):
        current = _collect_visible_reviews(driver)
        prev_len = len(seen)
        seen.update(current)

        if len(seen) >= target_count:
            break

        # 더보기 버튼
        clicked = False
        for el in driver.find_elements(By.CSS_SELECTOR,
                "button, a, [role='button'], [class*='MoreButton'], [class*='more-btn']"):
            try:
                t = el.text.strip()
                if el.is_displayed() and re.search(r"더\s*보기|more|More|전체", t):
                    safe_click(driver, el)
                    time.sleep(1.0)
                    clicked = True
                    break
            except (StaleElementReferenceException, WebDriverException):
                continue

        if not clicked:
            driver.execute_script("window.scrollBy(0, 700);")
            time.sleep(0.8)

        if len(seen) == prev_len:
            stuck += 1
            if stuck >= 3:
                break
        else:
            stuck = 0

    return list(seen)[:target_count]


# ── CSV I/O ───────────────────────────────────────────────────────────────────

def read_existing(path):
    if not path.exists():
        return []
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


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    _replace(tmp, path)


def read_statuses():
    if not STATUS_PATH.exists():
        return {}
    with STATUS_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        return {r["restaurant_id"]: r for r in csv.DictReader(f) if r.get("restaurant_id")}


def write_statuses(statuses):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_PATH.with_suffix(f".{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(statuses.values())
    _replace(tmp, STATUS_PATH)


def append_error(row):
    ERRORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = ERRORS_PATH.exists()
    with ERRORS_PATH.open("a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ERROR_FIELDNAMES)
        if not exists:
            w.writeheader()
        w.writerow(row)


def _replace(src, dst, attempts=5):
    for i in range(attempts):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.5)


def review_counts(rows):
    counts = {}
    for r in rows:
        rid = r.get("restaurant_id", "")
        if rid:
            counts[rid] = counts.get(rid, 0) + 1
    return counts


def status_label(collected, target):
    if collected >= target:
        return "completed"
    return "partial" if collected > 0 else "no_reviews"


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="망고플레이트 리뷰 크롤러")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--show-browser", action="store_false", dest="headless")
    parser.add_argument("--area", help="특정 지역만 크롤 (예: 강남)")
    parser.add_argument("--category", help="특정 카테고리만 크롤 (예: 한식)")
    args = parser.parse_args()

    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8-sig"))
    target = plan["reviews_per_restaurant"]
    limit = plan["restaurants_per_group"]

    areas = [args.area] if args.area else plan["areas"]
    categories = [args.category] if args.category else list(CATEGORY_SEARCH.keys())

    all_rows = dedupe(read_existing(OUTPUT_PATH))
    statuses = read_statuses()
    counts = review_counts(all_rows)

    if all_rows:
        print(f"[resume] 기존 리뷰 {len(all_rows)}개 로드 ({OUTPUT_PATH.name})")

    driver = make_driver(headless=args.headless)
    try:
        for area in areas:
            for category in categories:
                print(f"\n[search] {area} / {category}")
                try:
                    places = search_places(driver, area, category, limit)
                except (TimeoutException, WebDriverException) as e:
                    print(f"[error] 검색 실패: {e}")
                    append_error({"area": area, "category": category,
                                  "restaurant_id": "", "restaurant_name": "",
                                  "stage": "search", "error": repr(e)})
                    continue

                print(f"[found] {len(places)}개 식당")

                for idx, place in enumerate(places, 1):
                    if (counts.get(place.place_id, 0) >= target
                            or statuses.get(place.place_id, {}).get("status") == "completed"):
                        print(f"[skip] {idx}/{len(places)} {place.name}")
                        continue

                    print(f"[collect] {idx}/{len(places)} {place.name}")
                    try:
                        # 메타 정보 (별점, 리뷰수, 가격)
                        driver.get(place.url)
                        time.sleep(1.5)
                        rating, review_count, price = extract_meta(driver)
                        if rating:
                            place.rating = rating
                        if review_count:
                            place.review_count = review_count

                        # 리뷰 수집
                        reviews = collect_reviews(driver, place.url, target)
                    except (TimeoutException, WebDriverException, JavascriptException) as e:
                        print(f"[error] 수집 실패: {place.name} / {e}")
                        append_error({"area": area, "category": category,
                                      "restaurant_id": place.place_id,
                                      "restaurant_name": place.name,
                                      "stage": "collect", "error": repr(e)})
                        continue

                    new_rows = [
                        {
                            "restaurant_id": place.place_id,
                            "restaurant_name": place.name,
                            "area": area,
                            "category": category,
                            "rating": place.rating,
                            "review_count": place.review_count,
                            "price": price,
                            "photo_ratio": "",
                            "review_text": r,
                        }
                        for r in reviews
                    ]

                    all_rows = dedupe(all_rows + new_rows)
                    write_rows(OUTPUT_PATH, all_rows)
                    counts = review_counts(all_rows)

                    collected = counts.get(place.place_id, 0)
                    statuses[place.place_id] = {
                        "area": area, "category": category,
                        "restaurant_id": place.place_id,
                        "restaurant_name": place.name,
                        "status": status_label(collected, target),
                        "collected_count": collected,
                        "target_count": target,
                    }
                    write_statuses(statuses)
                    print(f"  → 리뷰 {collected}개 수집")
                    time.sleep(1.0)

    finally:
        driver.quit()

    print(f"\n완료. 저장: {OUTPUT_PATH}")
    print(f"총 리뷰: {len(all_rows)}개")


if __name__ == "__main__":
    main()
