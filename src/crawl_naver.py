"""
네이버 플레이스 방문자 리뷰 크롤러.

공개 페이지를 저속으로 수집하는 연구용 스크립트입니다. 실행 전 서비스 약관과
robots 정책을 확인하고, 과도한 요청을 피하세요.
"""

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

# 네이버 플레이스의 동적 렌더링을 안정적으로 처리하기 위해 Chrome 자동화를 사용합니다.
import undetected_chromedriver as uc
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    JavascriptException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "config" / "sampling_plan.json"
OUTPUT_PATH = ROOT / "data" / "raw_reviews_naver.csv"
STATUS_PATH = ROOT / "data" / "crawl_status_naver.csv"
ERRORS_PATH = ROOT / "data" / "crawl_errors_naver.csv"

SEARCH_URL = "https://map.naver.com/p/search/{query}"

FIELDNAMES = [
    "restaurant_id", "restaurant_name", "area", "category",
    "rating", "review_count", "price", "photo_ratio", "review_text",
]
STATUS_FIELDNAMES = [
    "area", "category", "restaurant_id", "restaurant_name",
    "status", "collected_count", "target_count",
]
ERROR_FIELDNAMES = ["area", "category", "restaurant_id", "restaurant_name", "stage", "error"]

CATEGORY_SEARCH = {
    "한식": "한식", "일식": "일식", "중식": "중식", "양식": "양식",
    "술집": "술집", "치킨": "치킨", "분식": "분식", "이자카야": "이자카야",
    "고깃집": "고깃집", "호프/통닭": "호프", "카페 디저트": "카페",
}


@dataclass
class Place:
    place_id: str
    name: str
    rating: str = ""
    review_count: str = ""
    element_index: int = 0  # 자연스러운 클릭 흐름 유지를 위해 인덱스 기록


# ── 드라이버 ───────────────────────────────────

def make_driver(headless=True, chrome_version=None):
    options = uc.ChromeOptions()
    
    if headless:
        # uc 환경에서는 --headless=new 대신 아래 인자가 안정적입니다.
        options.add_argument("--headless") 
    
    options.add_argument("--disable-gpu")
    options.add_argument("--lang=ko-KR,ko")
    options.add_argument("--window-size=1400,900")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    # 자동화 환경에서 렌더링이 흔들리는 경우를 줄이기 위한 옵션.
    options.add_argument("--disable-blink-features=AutomationControlled")

    version = chrome_version
    if version is None:
        env_version = os.environ.get("NAVER_CRAWLER_CHROME_VERSION")
        if env_version:
            version = int(env_version)

    kwargs = {"options": options}
    if version:
        kwargs["version_main"] = version
    return uc.Chrome(**kwargs)


def safe_click(driver, element):
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", element)
    time.sleep(0.4)
    try:
        element.click()
    except (ElementClickInterceptedException, StaleElementReferenceException):
        driver.execute_script("arguments[0].click();", element)


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def digits(text):
    m = re.search(r"[\d,]+", text or "")
    return m.group(0).replace(",", "") if m else ""


def switch_to_iframe(driver, selectors, timeout=10):
    wait = WebDriverWait(driver, timeout)
    for sel in selectors:
        try:
            frame = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, sel)))
            driver.switch_to.frame(frame)
            return True
        except TimeoutException:
            continue
    return False


# ── 검색 및 목록 추출 ─────────────────────────────────────────────────────────

def _extract_place_id_from_url(driver):
    m = re.search(r"/place/(\d+)", driver.current_url)
    if m:
        return m.group(1)
    driver.switch_to.default_content()
    frames = driver.find_elements(By.CSS_SELECTOR, "#entryIframe")
    if frames:
        src = frames[0].get_attribute("src") or ""
        m = re.search(r"/place/(\d+)", src)
        if m:
            return m.group(1)
    return None


def search_places(driver, area, category, limit):
    search_kw = CATEGORY_SEARCH.get(category, category)
    query = quote(f"{area} {search_kw} 맛집")
    driver.get(SEARCH_URL.format(query=query))
    time.sleep(3.0)

    driver.switch_to.default_content()
    if not switch_to_iframe(driver, ["#searchIframe", "iframe[src*='place/list']"]):
        print(f"  [warn] searchIframe 없음: {area}/{category}")
        return []

    wait = WebDriverWait(driver, 12)
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "li.UEzoS")))
    except TimeoutException:
        print(f"  [warn] 검색 결과 로드 실패: {area}/{category}")
        return []

    # 검색 목록 스크롤 다운
    for _ in range(8):
        cards = driver.find_elements(By.CSS_SELECTOR, "li.UEzoS")
        if len(cards) >= limit:
            break
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.0)

    total_cards = len(driver.find_elements(By.CSS_SELECTOR, "li.UEzoS"))
    places = []
    seen = set()

    for i in range(total_cards):
        if len(places) >= limit:
            break

        try:
            cards = driver.find_elements(By.CSS_SELECTOR, "li.UEzoS")
            if i >= len(cards): break
            card = cards[i]
        except Exception:
            break

        name_els = card.find_elements(By.CSS_SELECTOR, "span.TYaxT")
        if not name_els: continue
        name = name_els[0].text.strip()
        if not name: continue

        rating = ""
        review_count = ""
        try:
            for sel in [".orXYY", "[class*='starScore']", "[class*='rating']"]:
                els = card.find_elements(By.CSS_SELECTOR, sel)
                if els:
                    t = els[0].text.strip()
                    # 순수 별점(0.5~5.0)만 채택 — "방문자 리뷰 1,068" 같은 오염 차단
                    if re.match(r"^\d+(\.\d+)?$", t) and 0.5 <= float(t) <= 5.0:
                        rating = t
                        break
            mv_els = card.find_elements(By.CSS_SELECTOR, ".MVx6e, [class*='reviewCount']")
            for el in mv_els:
                d = digits(el.text)
                # 합리적 범위(1~500,000)만 리뷰 수로 인정 — place id 등 큰 숫자 오염 차단
                if d and 1 <= int(d) <= 500000:
                    review_count = d
                    break
        except StaleElementReferenceException:
            pass

        try:
            link = card.find_element(By.CSS_SELECTOR, "a.YTJkH, a.CtW3e")
            driver.execute_script("arguments[0].click();", link)
            time.sleep(1.5)
            raw_id = _extract_place_id_from_url(driver)
        except Exception:
            raw_id = None

        if not raw_id:
            try:
                driver.switch_to.default_content()
                switch_to_iframe(driver, ["#searchIframe"])
            except Exception: pass
            continue

        place_id = f"nv_{raw_id}"
        if place_id in seen:
            try:
                driver.switch_to.default_content()
                switch_to_iframe(driver, ["#searchIframe"])
            except Exception:
                pass
            continue

        seen.add(place_id)
        places.append(Place(place_id=place_id, name=name,
                            rating=rating, review_count=review_count, element_index=i))

        # searchIframe으로 복귀
        try:
            driver.switch_to.default_content()
            switch_to_iframe(driver, ["#searchIframe"])
            time.sleep(0.3)
        except Exception:
            pass

    return places


# ── 리뷰 수집 (자연스러운 전환 구조로 개편) ──────────────────────────────────

def _collect_visible_reviews(driver):
    # 최신 네이버 플레이스 타겟 셀렉터 추가 및 보강
    selectors = [
        ".pui__vn15t2",           # 리뷰 텍스트 컨테이너 (DOM 직접 확인)
        ".pui__vn15t2 span",      # 내부 span
        ".pui__xvbZA",
        "span.z6370",
        "span.flick-content",
        ".YEtaQ",
        "span[class*='place_review']",
        "li[class*='ReviewItem'] span",
        "[class*='review-content']",
    ]
    seen, texts = set(), []
    for sel in selectors:
        for el in driver.find_elements(By.CSS_SELECTOR, sel):
            try:
                t = el.text.strip()
            except StaleElementReferenceException:
                continue
            # "더보기" / "접기" 버튼 텍스트 제거
            t = re.sub(r"\s*(더보기|접기)\s*$", "", t).strip()
            if len(t) >= 5 and t not in seen:
                seen.add(t)
                texts.append(t)
    return texts


_plan_cache = {}


def _get_target_from_plan():
    if not _plan_cache:
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8-sig"))
        _plan_cache["target"] = plan["reviews_per_restaurant"]
    return _plan_cache["target"]


def collect_reviews_and_meta(driver, place, area, category):
    """placePath 직접 호출 대신, 플레이스 메인 페이지에서 방문자 리뷰 탭을 클릭해 자연스럽게 진입."""
    raw_id = place.place_id.replace("nv_", "")
    driver.switch_to.default_content()

    # placePath 없이 플레이스 메인 페이지로 진입 (봇 감지 최소화)
    nav_url = f"https://map.naver.com/p/search/{quote(place.name, safe='')}/place/{raw_id}"
    driver.get(nav_url)
    time.sleep(3.0)

    if not switch_to_iframe(driver, ["#entryIframe"]):
        return []

    # 콘텐츠 렌더링 대기
    def _content_loaded(d):
        return len([el for el in d.find_elements(By.CSS_SELECTOR, "span")
                    if len(el.text.strip()) >= 5]) > 0

    try:
        WebDriverWait(driver, 15).until(_content_loaded)
    except TimeoutException:
        pass
    time.sleep(0.5)

    # IP 차단 감지
    body_text = driver.execute_script("return document.body ? document.body.innerText : '';")
    if "이용이 제한" in body_text:
        ip_m = re.search(r"IP:\s*([\d.]+)", body_text)
        print(f"  [warn] IP 차단 감지{' (' + ip_m.group(1) + ')' if ip_m else ''}")
        return []

    # 방문자 리뷰 탭 자연스럽게 클릭
    clicked_tab = False
    for sel in [
        "a[href*='review/visitor']",
        "ul[class*='tab'] a",
        "[class*='tab'] a",
    ]:
        for tab in driver.find_elements(By.CSS_SELECTOR, sel):
            try:
                if "방문자" in tab.text:
                    safe_click(driver, tab)
                    time.sleep(1.5)
                    clicked_tab = True
                    break
            except (StaleElementReferenceException, WebDriverException):
                continue
        if clicked_tab:
            break

    time.sleep(1.0)

    # 별점 / 리뷰 수 갱신
    wait = WebDriverWait(driver, 10)
    try:
        rating_el = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,
            ".place_section .PXMot, [class*='starScore'], [class*='ratingValue']")))
        t = rating_el.text.strip()
        # 순수 숫자(소수점 포함)만 별점으로 인정 — "방문자 리뷰 1,068" 같은 오염 방지
        if re.match(r"^\d+(\.\d+)?$", t):
            place.rating = t
    except TimeoutException:
        pass

    for sel in ["[class*='ReviewCount']", "[class*='visitorReviewCount']"]:
        for el in driver.find_elements(By.CSS_SELECTOR, sel):
            d = digits(el.text)
            # 합리적 범위(1 ~ 500,000)만 리뷰 수로 인정 — 가격 등 오염 방지
            if d and 1 <= int(d) <= 500000:
                place.review_count = d
                break
        if place.review_count:
            break

    # 리뷰 무한 스크롤 수집
    seen: set = set()
    stuck = 0
    target = int(_get_target_from_plan())

    for _ in range(25):
        current = _collect_visible_reviews(driver)
        prev_len = len(seen)
        seen.update(current)

        if len(seen) >= target:
            break

        clicked = False
        for el in driver.find_elements(By.CSS_SELECTOR,
                "a.pui__GStJHb, a.fvwqf, button[class*='more'], [class*='moreBtn'], a[data-nclick*='more']"):
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
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)
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
    parser.add_argument("--limit", type=int, help="식당 수 제한 (테스트용)")
    parser.add_argument(
        "--chrome-version",
        type=int,
        help="Chrome major version override. 기본값은 undetected-chromedriver 자동 감지.",
    )
    args = parser.parse_args()

    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8-sig"))
    target = plan["reviews_per_restaurant"]
    limit = args.limit if args.limit else plan["restaurants_per_group"]

    areas = [args.area] if args.area else plan["areas"]
    categories = [args.category] if args.category else list(CATEGORY_SEARCH.keys())

    all_rows = dedupe(read_existing(OUTPUT_PATH))
    statuses = read_statuses()
    counts = review_counts(all_rows)

    if all_rows:
        print(f"[resume] 기존 리뷰 {len(all_rows)}개 로드")

    driver = make_driver(headless=args.headless, chrome_version=args.chrome_version)
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
                        reviews = collect_reviews_and_meta(driver, place, area, category)
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
                    time.sleep(1.5)

    finally:
        driver.quit()

    print(f"\n완료. 저장: {OUTPUT_PATH}")
    print(f"총 리뷰: {len(all_rows)}개")


if __name__ == "__main__":
    main()
