"""
네이버 플레이스 방문자 리뷰 크롤러

카카오맵보다 "데이트로 왔어요", "친구들이랑", "회식 자리" 같은
관계 맥락 표현이 많아 모델 positive sample 증가가 목적.

출력 포맷: raw_reviews.csv 와 동일
  restaurant_id, restaurant_name, area, category,
  rating, review_count, price, photo_ratio, review_text

결과 파일:
  data/raw_reviews_naver.csv        ← 수집 결과
  data/crawl_status_naver.csv       ← 진행 상태 (재시작용)
  data/crawl_errors_naver.csv       ← 에러 로그
"""

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

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
OUTPUT_PATH = ROOT / "data" / "raw_reviews_naver.csv"
STATUS_PATH = ROOT / "data" / "crawl_status_naver.csv"
ERRORS_PATH = ROOT / "data" / "crawl_errors_naver.csv"

# 네이버 플레이스 URL
SEARCH_URL = "https://map.naver.com/p/search/{query}"
REVIEW_URL = "https://pcmap.place.naver.com/restaurant/{place_id}/review/visitor"

FIELDNAMES = [
    "restaurant_id", "restaurant_name", "area", "category",
    "rating", "review_count", "price", "photo_ratio", "review_text",
]
STATUS_FIELDNAMES = [
    "area", "category", "restaurant_id", "restaurant_name",
    "status", "collected_count", "target_count",
]
ERROR_FIELDNAMES = ["area", "category", "restaurant_id", "restaurant_name", "stage", "error"]

# 카카오맵 카테고리 → 네이버 검색 키워드
CATEGORY_SEARCH = {
    "한식":     "한식",
    "일식":     "일식",
    "중식":     "중식",
    "양식":     "양식",
    "술집":     "술집",
    "치킨":     "치킨",
    "분식":     "분식",
    "이자카야":  "이자카야",
    "고깃집":   "고깃집",
    "호프/통닭": "호프",
    "카페 디저트": "카페",
}


@dataclass
class Place:
    place_id: str
    name: str
    rating: str = ""
    review_count: str = ""


# ── 드라이버 ──────────────────────────────────────────────────────────────────

def make_driver(headless=True):
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--lang=ko-KR,ko")
    options.add_argument("--window-size=1400,900")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver


def safe_click(driver, element):
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", element)
    time.sleep(0.3)
    try:
        element.click()
    except (ElementClickInterceptedException, StaleElementReferenceException):
        driver.execute_script("arguments[0].click();", element)


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def digits(text):
    m = re.search(r"[\d,]+", text or "")
    return m.group(0).replace(",", "") if m else ""


def switch_to_iframe(driver, selectors, timeout=10):
    """iframe 전환. 실패하면 False 반환."""
    wait = WebDriverWait(driver, timeout)
    for sel in selectors:
        try:
            frame = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, sel)))
            driver.switch_to.frame(frame)
            return True
        except TimeoutException:
            continue
    return False


# ── 검색: 식당 목록 수집 ─────────────────────────────────────────────────────

def search_places(driver, area, category, limit):
    search_kw = CATEGORY_SEARCH.get(category, category)
    query = quote(f"{area} {search_kw} 맛집")
    driver.get(SEARCH_URL.format(query=query))
    time.sleep(2.5)

    # 검색 결과 iframe 진입
    driver.switch_to.default_content()
    if not switch_to_iframe(driver, ["#searchIframe", "iframe[src*='place/list']"]):
        print(f"  [warn] searchIframe 없음: {area}/{category}")
        return []

    wait = WebDriverWait(driver, 12)
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,
            "li.UEzoS, li[class*='PlaceItem'], .place_bluelink, li[data-nclick*='plc']")))
    except TimeoutException:
        print(f"  [warn] 검색 결과 로드 실패: {area}/{category}")
        return []

    places = []
    seen = set()

    for _ in range(10):
        cards = driver.find_elements(By.CSS_SELECTOR,
            "li.UEzoS, li[class*='PlaceItem'], li[data-nclick*='plc']")

        for card in cards:
            # 식당명
            name = ""
            for sel in ["span.place_bluelink", ".TYaxT", "a[class*='name']", "span[class*='name']"]:
                els = card.find_elements(By.CSS_SELECTOR, sel)
                if els:
                    name = els[0].text.strip()
                    if name:
                        break

            # place_id: href에서 추출
            place_id = ""
            for a in card.find_elements(By.CSS_SELECTOR, "a"):
                href = a.get_attribute("href") or ""
                m = re.search(r"/(?:entry/place|p/entry/restaurant|restaurant)/(\d+)", href)
                if not m:
                    # data 속성에서 시도
                    nclick = a.get_attribute("data-nclick") or ""
                    m2 = re.search(r"plc\.(\d+)", nclick)
                    if m2:
                        place_id = f"nv_{m2.group(1)}"
                        break
                else:
                    place_id = f"nv_{m.group(1)}"
                    break

            if not place_id:
                # li 자체의 data 속성
                for attr in ["data-id", "data-place-id"]:
                    val = card.get_attribute(attr) or ""
                    if val.isdigit():
                        place_id = f"nv_{val}"
                        break

            if not name or not place_id or place_id in seen:
                continue
            seen.add(place_id)

            # 별점, 리뷰 수
            rating = ""
            for sel in [".orXYY", "[class*='rating']", "[class*='score']"]:
                els = card.find_elements(By.CSS_SELECTOR, sel)
                if els:
                    t = els[0].text.strip()
                    if re.search(r"\d", t):
                        rating = t
                        break

            review_count = ""
            for sel in [".MVx6e", "[class*='review']", "[class*='count']"]:
                els = card.find_elements(By.CSS_SELECTOR, sel)
                if els:
                    review_count = digits(els[0].text)
                    if review_count:
                        break

            places.append(Place(place_id=place_id, name=name,
                                rating=rating, review_count=review_count))
            if len(places) >= limit:
                return places

        # 더보기 / 스크롤
        more = driver.find_elements(By.CSS_SELECTOR,
            "a.fvwqf, button[class*='more'], a[class*='more'], .BXNBd a")
        clicked = False
        for btn in more:
            if btn.is_displayed():
                try:
                    safe_click(driver, btn)
                    time.sleep(1.5)
                    clicked = True
                    break
                except Exception:
                    pass
        if not clicked:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.2)

        new_cards = driver.find_elements(By.CSS_SELECTOR,
            "li.UEzoS, li[class*='PlaceItem'], li[data-nclick*='plc']")
        if len(new_cards) <= len(cards) and not clicked:
            break

    return places


# ── 리뷰 수집 ─────────────────────────────────────────────────────────────────

def _get_raw_place_id(place_id):
    return place_id.replace("nv_", "")


def _collect_visible_reviews(driver):
    selectors = [
        ".pui__xvbZA",           # 방문자 리뷰 텍스트 (주요)
        ".YEtaQ",
        "span[class*='place_review']",
        ".place_section .pui__vn15t2",
        "li[class*='ReviewItem'] span",
        "[class*='review-content']",
        "[class*='reviewArea'] span",
    ]
    seen, texts = set(), []
    for sel in selectors:
        for el in driver.find_elements(By.CSS_SELECTOR, sel):
            t = el.text.strip()
            if len(t) >= 5 and t not in seen:
                seen.add(t)
                texts.append(t)
    return texts


def collect_reviews_and_meta(driver, place):
    raw_id = _get_raw_place_id(place.place_id)
    url = REVIEW_URL.format(place_id=raw_id)
    driver.get(url)
    time.sleep(2.0)

    wait = WebDriverWait(driver, 12)

    # 별점/리뷰수 갱신 시도 (place 페이지에서 더 정확)
    try:
        rating_el = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,
            ".place_section .PXMot, [class*='starScore'], [class*='ratingValue']")))
        t = rating_el.text.strip()
        if re.search(r"\d", t):
            place.rating = t
    except TimeoutException:
        pass

    for sel in ["[class*='ReviewCount']", "[class*='visitorReviewCount']", ".place_section em"]:
        els = driver.find_elements(By.CSS_SELECTOR, sel)
        for el in els:
            d = digits(el.text)
            if d:
                place.review_count = d
                break
        if place.review_count:
            break

    # 리뷰 무한 스크롤 수집
    seen = set()
    stuck = 0
    target = int(_get_target_from_plan())

    for _ in range(25):
        current = _collect_visible_reviews(driver)
        prev_len = len(seen)
        seen.update(current)

        if len(seen) >= target:
            break

        # 더보기 버튼
        clicked = False
        for el in driver.find_elements(By.CSS_SELECTOR,
                "a.fvwqf, button[class*='more'], [class*='moreBtn'], a[data-nclick*='more']"):
            try:
                t = el.text.strip()
                if el.is_displayed() and re.search(r"더\s*보기|more|More", t, re.I):
                    safe_click(driver, el)
                    time.sleep(1.2)
                    clicked = True
                    break
            except (StaleElementReferenceException, WebDriverException):
                continue

        if not clicked:
            driver.execute_script("window.scrollBy(0, 800);")
            time.sleep(0.9)

        if len(seen) == prev_len:
            stuck += 1
            if stuck >= 3:
                break
        else:
            stuck = 0

    return list(seen)[:target]


_plan_cache = {}


def _get_target_from_plan():
    if not _plan_cache:
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8-sig"))
        _plan_cache["target"] = plan["reviews_per_restaurant"]
    return _plan_cache["target"]


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
    tmp = path.with_name(f"{path.stem}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerows(rows)
    _replace(tmp, path)


def read_statuses():
    if not STATUS_PATH.exists():
        return {}
    with STATUS_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        return {r["restaurant_id"]: r for r in csv.DictReader(f) if r.get("restaurant_id")}


def write_statuses(statuses):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_PATH.with_name(f"{STATUS_PATH.stem}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STATUS_FIELDNAMES)
        w.writeheader()
        w.writerows(statuses.values())
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
    parser = argparse.ArgumentParser(description="네이버 플레이스 방문자 리뷰 크롤러")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--show-browser", action="store_false", dest="headless")
    parser.add_argument("--area", help="특정 지역만 (예: 강남)")
    parser.add_argument("--category", help="특정 카테고리만 (예: 한식)")
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
        print(f"[resume] 기존 리뷰 {len(all_rows)}개 로드")

    driver = make_driver(headless=args.headless)
    try:
        for area in areas:
            for category in categories:
                print(f"\n[search] {area} / {category}")
                try:
                    places = search_places(driver, area, category, limit)
                    driver.switch_to.default_content()
                except (TimeoutException, WebDriverException) as e:
                    print(f"[error] 검색 실패: {e}")
                    driver.switch_to.default_content()
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

                    print(f"[collect] {idx}/{len(places)} {place.name} ({place.place_id})")
                    try:
                        reviews = collect_reviews_and_meta(driver, place)
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
                            "price": "",
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
                    time.sleep(1.2)

    finally:
        driver.quit()

    print(f"\n완료. 저장: {OUTPUT_PATH}")
    print(f"총 리뷰: {len(all_rows)}개")


if __name__ == "__main__":
    main()
