import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "config" / "sampling_plan.json"
ERRORS_PATH = ROOT / "data" / "crawl_errors.csv"
STATUS_PATH = ROOT / "data" / "crawl_status.csv"

FIELDNAMES = [
    "restaurant_id",
    "restaurant_name",
    "area",
    "category",
    "rating",
    "review_count",
    "price",
    "photo_ratio",
    "review_text",
]

ERROR_FIELDNAMES = ["area", "category", "restaurant_id", "restaurant_name", "stage", "error"]
STATUS_FIELDNAMES = [
    "area",
    "category",
    "restaurant_id",
    "restaurant_name",
    "status",
    "collected_count",
    "target_count",
]

MORE_WORDS = ["\ub354\ubcf4\uae30", "\ud6c4\uae30 \ub354\ubcf4\uae30", "\ub9ac\ubdf0 \ub354\ubcf4\uae30"]


@dataclass
class Place:
    place_id: str
    name: str
    url: str
    rating: str = ""
    review_count: str = ""


def load_plan():
    with PLAN_PATH.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def make_driver(headless=False):
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--lang=ko-KR")
    options.add_argument("--window-size=1400,1000")
    return webdriver.Chrome(options=options)


def safe_click(driver, element):
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
    time.sleep(0.3)
    try:
        element.click()
    except Exception:
        driver.execute_script("arguments[0].click();", element)


def text_or_empty(element, selectors):
    for selector in selectors:
        found = element.find_elements(By.CSS_SELECTOR, selector)
        if found:
            return found[0].text.strip()
    return ""


def digits(text):
    match = re.search(r"[\d,]+", text or "")
    return match.group(0).replace(",", "") if match else ""


def extract_place_id(url):
    match = re.search(r"/(\d+)(?:\?|$)", url)
    if match:
        return match.group(1)
    return re.sub(r"\W+", "_", url)[-40:]


def search_places(driver, area, category, limit):
    query = f"{area} {category} \ub9db\uc9d1"
    url = f"https://map.kakao.com/?q={quote_plus(query)}"
    driver.get(url)
    wait = WebDriverWait(driver, 12)

    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "ul.placelist, .placelist")))
    except TimeoutException:
        print(f"[warn] no search results loaded: {query}")
        return []

    places = []
    seen = set()

    for _ in range(8):
        cards = driver.find_elements(By.CSS_SELECTOR, "li.PlaceItem, li[data-id], .placelist > li")
        for card in cards:
            name = text_or_empty(card, [".link_name", ".tit_name", "a[data-id='name']", "strong"])
            link = None
            for selector in ["a.moreview", "a.link_name", "a[href*='/place/']"]:
                links = card.find_elements(By.CSS_SELECTOR, selector)
                if links:
                    link = links[0].get_attribute("href")
                    break
            if not name or not link:
                continue
            place_id = extract_place_id(link)
            if place_id in seen:
                continue
            seen.add(place_id)
            rating = text_or_empty(card, [".score .num", ".rating", ".num"])
            review_count = digits(text_or_empty(card, [".review", ".count", ".numberofscore"]))
            places.append(Place(place_id=place_id, name=name, url=link, rating=rating, review_count=review_count))
            if len(places) >= limit:
                return places

        more_buttons = driver.find_elements(By.CSS_SELECTOR, "a#info.search.place.more, .more")
        clickable = [button for button in more_buttons if button.is_displayed()]
        if not clickable:
            break
        safe_click(driver, clickable[0])
        time.sleep(1.2)

    return places


def click_review_tab(driver):
    selectors = [
        "a[href*='review']",
        "button[aria-controls*='review']",
        ".link_tab[href*='comment']",
    ]
    for selector in selectors:
        buttons = driver.find_elements(By.CSS_SELECTOR, selector)
        for button in buttons:
            if button.is_displayed():
                try:
                    safe_click(driver, button)
                    time.sleep(0.8)
                    return
                except Exception:
                    pass


def expand_reviews(driver, target_count):
    last_count = 0
    stuck_rounds = 0

    for _ in range(10):
        reviews = collect_review_texts(driver)
        print(f"  collected visible reviews: {len(reviews)}")
        if len(reviews) >= target_count:
            print(f"  reached target reviews: {len(reviews)}")
            return

        if len(reviews) == last_count:
            stuck_rounds += 1
        else:
            stuck_rounds = 0
            last_count = len(reviews)

        if stuck_rounds >= 2:
            print(f"  no more reviews loaded; moving on with {len(reviews)} reviews")
            return

        driver.execute_script("window.scrollBy(0, Math.floor(window.innerHeight * 1.2));")
        time.sleep(0.7)


def collect_review_texts(driver):
    selectors = [
        ".list_evaluation .txt_comment",
        ".list_review .desc_review",
        "p[class*='txt_']",
        "div[class*='comment'] p",
        "[class*='review'] [class*='txt']",
        "[class*='comment']",
    ]
    texts = []
    seen = set()
    for selector in selectors:
        for element in driver.find_elements(By.CSS_SELECTOR, selector):
            text = element.text.strip()
            if len(text) < 3 or text in seen:
                continue
            seen.add(text)
            texts.append(text)
    return texts


def extract_price(driver):
    """Extract average price from the place info page.

    Tries multiple selectors that Kakao Map uses for price/menu info.
    Returns a price string (digits only) or empty string if not found.
    """
    price_selectors = [
        ".info_price .txt_price",
        "[class*='price'] [class*='txt']",
        ".detail_price",
        ".info_menu .price_menu",
        "[class*='menu'] [class*='price']",
        ".list_menu .price_menu",
    ]
    prices = []
    for selector in price_selectors:
        for element in driver.find_elements(By.CSS_SELECTOR, selector):
            text = element.text.strip()
            match = re.search(r"[\d,]+", text)
            if match:
                value = int(match.group(0).replace(",", ""))
                if 1000 <= value <= 500000:
                    prices.append(value)
    if prices:
        avg = sum(prices) // len(prices)
        return str(avg)
    return ""


def extract_photo_ratio(driver):
    """Estimate photo review ratio from visible review elements.

    Counts review items that contain image elements vs total review items.
    Returns a ratio string (0.0~1.0) or empty string if not calculable.
    """
    review_container_selectors = [
        ".list_evaluation > li",
        ".list_review > li",
        "[class*='review_list'] > li",
        "[class*='review'] > [class*='item']",
    ]
    total = 0
    with_photo = 0
    for selector in review_container_selectors:
        items = driver.find_elements(By.CSS_SELECTOR, selector)
        if not items:
            continue
        total = len(items)
        for item in items:
            photos = item.find_elements(By.CSS_SELECTOR, "img[src*='photo'], img[src*='img'], .photo_area img")
            if photos:
                with_photo += 1
        break

    if total >= 3:
        ratio = round(with_photo / total, 2)
        return str(ratio)
    return ""


def collect_place_info(driver, place_url):
    """Navigate to the place info page and extract price and photo_ratio.

    Visits the main info tab first, then the review tab.
    Returns (price, photo_ratio) as strings.
    """
    info_url = place_url.split("#")[0]
    driver.get(info_url)
    time.sleep(1.5)

    price = extract_price(driver)
    if price:
        print(f"  extracted price: {price}")

    return price


def collect_place_reviews(driver, place, area, category, review_limit):
    price = collect_place_info(driver, place.url)

    url = place.url.split("#")[0] + "#review"
    driver.get(url)
    time.sleep(1.5)
    if "#review" not in driver.current_url:
        click_review_tab(driver)

    photo_ratio = extract_photo_ratio(driver)
    if photo_ratio:
        print(f"  extracted photo_ratio: {photo_ratio}")

    expand_reviews(driver, review_limit)
    reviews = collect_review_texts(driver)[:review_limit]

    if not reviews:
        print(f"[warn] no reviews collected: {place.name}")

    rows = []
    for review in reviews:
        rows.append(
            {
                "restaurant_id": place.place_id,
                "restaurant_name": place.name,
                "area": area,
                "category": category,
                "rating": place.rating,
                "review_count": place.review_count,
                "price": price,
                "photo_ratio": photo_ratio,
                "review_text": review,
            }
        )
    return rows


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def append_error(row):
    ERRORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = ERRORS_PATH.exists()
    with ERRORS_PATH.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ERROR_FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_statuses():
    if not STATUS_PATH.exists():
        return {}
    with STATUS_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["restaurant_id"]: row for row in csv.DictReader(f) if row.get("restaurant_id")}


def write_statuses(statuses):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = STATUS_PATH.with_suffix(STATUS_PATH.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(statuses.values())
    temp_path.replace(STATUS_PATH)


def status_for_count(collected_count, target_count):
    if collected_count >= target_count:
        return "completed"
    if collected_count > 0:
        return "partial"
    return "no_reviews"


def is_finished_status(status):
    return status in {"completed", "partial", "no_reviews"}


def read_existing_rows(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def dedupe_rows(rows):
    deduped = []
    seen = set()
    for row in rows:
        key = (row.get("restaurant_id", ""), row.get("review_text", ""))
        if not key[0] or not key[1] or key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def review_counts_by_restaurant(rows):
    counts = {}
    for row in rows:
        restaurant_id = row.get("restaurant_id", "")
        if restaurant_id:
            counts[restaurant_id] = counts.get(restaurant_id, 0) + 1
    return counts


def main():
    plan = load_plan()
    output_path = ROOT / plan["output_path"]
    all_rows = dedupe_rows(read_existing_rows(output_path))
    statuses = read_statuses()
    if all_rows:
        print(f"[resume] loaded {len(all_rows)} existing reviews from {output_path}")
    if statuses:
        print(f"[resume] loaded {len(statuses)} crawl statuses from {STATUS_PATH}")

    driver = make_driver(headless=False)
    try:
        for area in plan["areas"]:
            for category in plan["categories"]:
                print(f"[search] {area} / {category}")
                try:
                    places = search_places(driver, area, category, plan["restaurants_per_group"])
                except Exception as error:
                    print(f"[error] search failed: {area} / {category} / {error}")
                    append_error(
                        {
                            "area": area,
                            "category": category,
                            "restaurant_id": "",
                            "restaurant_name": "",
                            "stage": "search",
                            "error": repr(error),
                        }
                    )
                    continue

                print(f"[found] {len(places)} places")
                for idx, place in enumerate(places, start=1):
                    existing_counts = review_counts_by_restaurant(all_rows)
                    previous_status = statuses.get(place.place_id, {}).get("status", "")
                    if (
                        existing_counts.get(place.place_id, 0) >= plan["reviews_per_restaurant"]
                        or is_finished_status(previous_status)
                    ):
                        print(
                            f"[skip] {area} {category} {idx}/{len(places)} "
                            f"{place.name} already tried ({previous_status or existing_counts.get(place.place_id, 0)})"
                        )
                        continue

                    print(f"[reviews] {area} {category} {idx}/{len(places)} {place.name}")
                    try:
                        rows = collect_place_reviews(driver, place, area, category, plan["reviews_per_restaurant"])
                    except Exception as error:
                        print(f"[error] reviews failed: {place.name} / {error}")
                        append_error(
                            {
                                "area": area,
                                "category": category,
                                "restaurant_id": place.place_id,
                                "restaurant_name": place.name,
                                "stage": "reviews",
                                "error": repr(error),
                            }
                        )
                        continue

                    all_rows = dedupe_rows(all_rows + rows)
                    write_rows(output_path, all_rows)
                    collected_count = review_counts_by_restaurant(all_rows).get(place.place_id, 0)
                    statuses[place.place_id] = {
                        "area": area,
                        "category": category,
                        "restaurant_id": place.place_id,
                        "restaurant_name": place.name,
                        "status": status_for_count(collected_count, plan["reviews_per_restaurant"]),
                        "collected_count": collected_count,
                        "target_count": plan["reviews_per_restaurant"],
                    }
                    write_statuses(statuses)
                    time.sleep(1.0)
    finally:
        driver.quit()

    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
