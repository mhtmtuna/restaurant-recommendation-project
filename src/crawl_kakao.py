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
    query = f"{area} {category} 맛집"
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
        clickable[0].click()
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
                    button.click()
                    time.sleep(0.8)
                    return
                except Exception:
                    pass


def expand_reviews(driver, target_count):
    for _ in range(10):
        reviews = collect_review_texts(driver)
        if len(reviews) >= target_count:
            return
        buttons = driver.find_elements(By.CSS_SELECTOR, "button, a")
        more = [
            button
            for button in buttons
            if button.is_displayed() and any(word in button.text for word in ["더보기", "후기 더보기", "리뷰 더보기"])
        ]
        if not more:
            return
        try:
            more[0].click()
            time.sleep(0.8)
        except Exception:
            return


def collect_review_texts(driver):
    selectors = [
        ".list_evaluation .txt_comment",
        ".list_review .desc_review",
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


def collect_place_reviews(driver, place, area, category, review_limit):
    driver.get(place.url)
    time.sleep(1.5)
    click_review_tab(driver)
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
                "price": "",
                "photo_ratio": "",
                "review_text": review,
            }
        )
    return rows


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main():
    plan = load_plan()
    output_path = ROOT / plan["output_path"]
    all_rows = []

    driver = make_driver(headless=False)
    try:
        for area in plan["areas"]:
            for category in plan["categories"]:
                print(f"[search] {area} / {category}")
                places = search_places(driver, area, category, plan["restaurants_per_group"])
                print(f"[found] {len(places)} places")
                for idx, place in enumerate(places, start=1):
                    print(f"[reviews] {area} {category} {idx}/{len(places)} {place.name}")
                    rows = collect_place_reviews(driver, place, area, category, plan["reviews_per_restaurant"])
                    all_rows.extend(rows)
                    write_rows(output_path, all_rows)
                    time.sleep(1.0)
    finally:
        driver.quit()

    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
