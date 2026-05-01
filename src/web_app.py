import csv
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
SCORES_PATH = ROOT / "data" / "restaurant_label_scores.csv"
FEATURES_PATH = ROOT / "data" / "restaurants_features.csv"

LABEL_COLUMNS = {
    "연인_식사": "couple_meal_score",
    "연인_술자리": "couple_drink_score",
    "친구_식사": "friend_meal_score",
    "친구_술자리": "friend_drink_score",
    "비즈니스_식사": "business_meal_score",
    "비즈니스_술자리": "business_drink_score",
}


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value, default=0.0):
    try:
        if value in ("", None):
            return default
        return float(value)
    except ValueError:
        return default


def load_restaurants():
    scores = read_csv(SCORES_PATH)
    features = {row["restaurant_id"]: row for row in read_csv(FEATURES_PATH)}
    restaurants = []

    for row in scores:
        feature = features.get(row["restaurant_id"], {})
        item = {
            "restaurant_id": row["restaurant_id"],
            "restaurant_name": row["restaurant_name"],
            "area": row["area"],
            "category": row["category"],
            "rating": to_float(row.get("rating")),
            "review_count": int(to_float(row.get("review_count"))),
            "seat_type": feature.get("seat_type", ""),
            "taste_score": to_float(feature.get("taste_score")),
            "value_score": to_float(feature.get("value_score")),
            "noise_score": to_float(feature.get("noise_score"), 0.5),
            "spaciousness_score": to_float(feature.get("spaciousness_score"), 0.5),
            "collected_review_count": int(to_float(feature.get("collected_review_count"))),
        }
        for label, column in LABEL_COLUMNS.items():
            item[column] = to_float(row.get(column))
        restaurants.append(item)

    return restaurants


HTML = r"""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>장소 추천 ML 시스템</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8f5;
      --ink: #1f2523;
      --muted: #65706b;
      --line: #d9ddd5;
      --panel: #ffffff;
      --accent: #1e7560;
      --accent-weak: #e1f0eb;
      --warm: #b45f35;
      --shadow: 0 10px 30px rgba(31, 37, 35, 0.08);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, "Malgun Gothic", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }

    header {
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,0.9);
      position: sticky;
      top: 0;
      z-index: 2;
    }

    .bar {
      max-width: 1180px;
      margin: 0 auto;
      padding: 16px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }

    .brand {
      font-size: 18px;
      font-weight: 800;
    }

    .meta {
      color: var(--muted);
      font-size: 13px;
    }

    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 24px 48px;
      display: grid;
      grid-template-columns: 380px 1fr;
      gap: 24px;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    .controls {
      padding: 20px;
      align-self: start;
      position: sticky;
      top: 78px;
    }

    h1 {
      font-size: 24px;
      margin: 0 0 8px;
      letter-spacing: 0;
    }

    p {
      margin: 0;
      line-height: 1.55;
    }

    .hint {
      color: var(--muted);
      font-size: 14px;
      margin-bottom: 18px;
    }

    textarea {
      width: 100%;
      min-height: 96px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      font: inherit;
      line-height: 1.45;
      outline: none;
      background: #fbfcfa;
    }

    textarea:focus,
    select:focus,
    input:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-weak);
    }

    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 12px;
    }

    label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin: 0 0 6px;
      font-weight: 700;
    }

    select,
    input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 11px;
      background: #fbfcfa;
      color: var(--ink);
      font: inherit;
      min-height: 42px;
    }

    .actions {
      display: flex;
      gap: 10px;
      margin-top: 16px;
    }

    button {
      border: 0;
      border-radius: 8px;
      padding: 11px 14px;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
      min-height: 42px;
    }

    .primary {
      background: var(--accent);
      color: white;
      flex: 1;
    }

    .secondary {
      background: var(--accent-weak);
      color: var(--accent);
    }

    .chips {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 12px;
    }

    .chip {
      border: 1px solid var(--line);
      background: #fbfcfa;
      color: var(--muted);
      border-radius: 999px;
      padding: 6px 9px;
      font-size: 12px;
    }

    .results {
      min-width: 0;
    }

    .summary {
      padding: 18px 20px;
      margin-bottom: 14px;
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: center;
    }

    .summary strong {
      display: block;
      font-size: 19px;
      margin-bottom: 4px;
    }

    .list {
      display: grid;
      gap: 12px;
    }

    .item {
      padding: 18px 20px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }

    .item-head {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 14px;
    }

    .rank {
      color: var(--accent);
      font-size: 13px;
      font-weight: 800;
    }

    .name {
      font-size: 18px;
      font-weight: 800;
      margin-top: 2px;
    }

    .score {
      font-size: 22px;
      font-weight: 900;
      color: var(--warm);
      white-space: nowrap;
    }

    .details {
      margin-top: 12px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    .detail {
      background: #f3f5f1;
      border-radius: 999px;
      padding: 6px 9px;
      color: var(--muted);
      font-size: 12px;
    }

    .empty {
      padding: 28px;
      text-align: center;
      color: var(--muted);
    }

    @media (max-width: 860px) {
      main {
        grid-template-columns: 1fr;
      }
      .controls {
        position: static;
      }
    }
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <div class="brand">장소 추천 ML 시스템</div>
      <div class="meta"><span id="restaurantCount"></span>개 식당 기반</div>
    </div>
  </header>

  <main>
    <section class="panel controls">
      <h1>어떤 자리야?</h1>
      <p class="hint">문장으로 적으면 아래 조건이 자동으로 채워져. 틀린 부분은 직접 고치면 돼.</p>

      <label for="natural">자연어 입력</label>
      <textarea id="natural">나는 남자 친한친구 3명이랑 해서 총 남자 4명이서 잠실에서 저녁 먹고 싶은데 추천해줘</textarea>

      <div class="actions">
        <button class="secondary" id="parseBtn">문장 분석</button>
        <button class="primary" id="recommendBtn">추천 보기</button>
      </div>

      <div class="chips" id="parsedChips"></div>

      <div class="row">
        <div>
          <label for="area">지역</label>
          <select id="area">
            <option>강남</option>
            <option>건대</option>
            <option>잠실</option>
          </select>
        </div>
        <div>
          <label for="relation">관계</label>
          <select id="relation">
            <option>친구</option>
            <option>연인</option>
            <option>비즈니스</option>
          </select>
        </div>
      </div>

      <div class="row">
        <div>
          <label for="occasion">상황</label>
          <select id="occasion">
            <option>식사</option>
            <option>술자리</option>
          </select>
        </div>
        <div>
          <label for="partySize">인원 수</label>
          <input id="partySize" type="number" min="1" max="20" value="4" />
        </div>
      </div>

      <div class="row">
        <div>
          <label for="category">음식 종류</label>
          <select id="category">
            <option>상관없음</option>
            <option>한식</option>
            <option>일식</option>
            <option>중식</option>
            <option>양식</option>
            <option>카페 디저트</option>
            <option>주점 바</option>
          </select>
        </div>
        <div>
          <label for="priority">우선 조건</label>
          <select id="priority">
            <option>균형</option>
            <option>맛</option>
            <option>가성비</option>
            <option>분위기</option>
          </select>
        </div>
      </div>

      <div class="row">
        <div>
          <label for="genderMix">성비</label>
          <select id="genderMix">
            <option>상관없음</option>
            <option>남자 위주</option>
            <option>여자 위주</option>
            <option>반반</option>
          </select>
        </div>
        <div>
          <label for="atmosphere">분위기</label>
          <select id="atmosphere">
            <option>상관없음</option>
            <option>조용</option>
            <option>활기</option>
            <option>넓은 곳</option>
          </select>
        </div>
      </div>
    </section>

    <section class="results">
      <div class="panel summary">
        <div>
          <strong id="summaryTitle">추천 결과</strong>
          <p class="hint" id="summaryText">조건을 분석한 뒤 추천을 보여줄게.</p>
        </div>
      </div>
      <div class="list" id="resultList"></div>
    </section>
  </main>

  <script>
    const restaurants = __RESTAURANTS__;
    const labelMap = {
      "연인_식사": "couple_meal_score",
      "연인_술자리": "couple_drink_score",
      "친구_식사": "friend_meal_score",
      "친구_술자리": "friend_drink_score",
      "비즈니스_식사": "business_meal_score",
      "비즈니스_술자리": "business_drink_score",
    };

    const $ = (id) => document.getElementById(id);

    function includesAny(text, words) {
      return words.some((word) => text.includes(word));
    }

    function parsePartySize(text) {
  const totalMatch = text.match(/총\s*(\d+)\s*명/);
  if (totalMatch) return Number(totalMatch[1]);

  // "남자 2에 여자 3명" 같은 패턴: 숫자들을 합산
  const genderPattern = text.match(/남자\s*(\d+).*여자\s*(\d+)|여자\s*(\d+).*남자\s*(\d+)/);
  if (genderPattern) {
    const nums = genderPattern.slice(1).filter(Boolean).map(Number);
    return nums.reduce((a, b) => a + b, 0);
  }

  const numberMatches = [...text.matchAll(/(\d+)\s*명/g)].map((m) => Number(m[1]));
  if (numberMatches.length > 1) return numberMatches.reduce((a, b) => a + b, 0);
  if (numberMatches.length === 1) return numberMatches[0];

      const wordNumbers = [
        ["둘", 2], ["두 명", 2], ["셋", 3], ["세 명", 3],
        ["넷", 4], ["네 명", 4], ["다섯", 5], ["여섯", 6],
      ];
      for (const [word, number] of wordNumbers) {
        if (text.includes(word)) return number;
      }
      return null;
    }

    function parseNatural(text) {
      const parsed = {};

      if (text.includes("강남")) parsed.area = "강남";
      if (text.includes("건대")) parsed.area = "건대";
      if (text.includes("잠실")) parsed.area = "잠실";

      let romanticMentioned = false;
      if (includesAny(text, ["여자친구", "남자친구", "연인", "데이트", "애인"])) {
        parsed.relation = "연인";
        parsed.partySize = 2;
        romanticMentioned = true;
      } else if (includesAny(text, ["회사", "상사", "팀장", "미팅", "비즈니스", "회식", "거래처"])) {
        parsed.relation = "비즈니스";
      } else if (includesAny(text, ["친구", "친한친구", "동기", "애들", "친구들"])) {
        parsed.relation = "친구";
      }

      if (includesAny(text, ["술", "한잔", "맥주", "소주", "와인", "칵테일", "2차"])) {
        parsed.occasion = "술자리";
      } else if (includesAny(text, ["밥", "식사", "저녁", "점심", "먹", "식당", "맛집"])) {
        parsed.occasion = "식사";
      }

      const partySize = parsePartySize(text);
      if (partySize && !romanticMentioned) parsed.partySize = partySize;

      if (includesAny(text, ["한식", "국밥", "고기", "삼겹살", "곱창", "찌개", "해장"])) {
        parsed.category = "한식";
      } else if (includesAny(text, ["일식", "초밥", "스시", "라멘", "이자카야", "사시미", "오마카세"])) {
        parsed.category = "일식";
      } else if (includesAny(text, ["중식", "짜장", "짬뽕", "마라", "훠궈", "양꼬치", "탕수육"])) {
        parsed.category = "중식";
      } else if (includesAny(text, ["양식", "파스타", "스테이크", "피자", "브런치", "버거"])) {
        parsed.category = "양식";
      } else if (includesAny(text, ["카페", "디저트", "커피", "케이크", "빙수", "베이커리"])) {
        parsed.category = "카페 디저트";
      } else if (includesAny(text, ["술집", "주점", "바", "포차", "맥주집", "와인바"])) {
        parsed.category = "주점 바";
      }

      if (includesAny(text, ["맛있는", "맛있", "맛집", "잘하는"])) parsed.priority = "맛";
      else if (includesAny(text, ["가성비", "저렴", "싸", "부담없는"])) parsed.priority = "가성비";
      else if (includesAny(text, ["분위기", "감성", "무드", "예쁜"])) parsed.priority = "분위기";

      const genderText = text.replaceAll("남자친구", "").replaceAll("여자친구", "");
      const maleMatch = genderText.match(/남자\s*(\d+)/);
      const femaleMatch = genderText.match(/여자\s*(\d+)/);
      if (maleMatch && femaleMatch) {
        const m = Number(maleMatch[1]);
        const f = Number(femaleMatch[1]);
        if (m > f) parsed.genderMix = "남자 위주";
        else if (f > m) parsed.genderMix = "여자 위주";
        else parsed.genderMix = "반반";
      } else if (hasMale) parsed.genderMix = "남자 위주";
      else if (hasFemale) parsed.genderMix = "여자 위주";
      else if (hasMale) parsed.genderMix = "남자 위주";
      else if (hasFemale) parsed.genderMix = "여자 위주";

      if (includesAny(text, ["조용", "대화", "차분"])) parsed.atmosphere = "조용";
      else if (includesAny(text, ["시끌", "활기", "핫플", "북적"])) parsed.atmosphere = "활기";
      else if (includesAny(text, ["넓", "단체", "회식", "자리 많"])) parsed.atmosphere = "넓은 곳";

      return parsed;
    }

    function applyParsed(parsed) {
      if (parsed.area) $("area").value = parsed.area;
      if (parsed.relation) $("relation").value = parsed.relation;
      if (parsed.occasion) $("occasion").value = parsed.occasion;
      if (parsed.partySize) $("partySize").value = parsed.partySize;
      if (parsed.category) $("category").value = parsed.category;
      if (parsed.priority) $("priority").value = parsed.priority;
      if (parsed.genderMix) $("genderMix").value = parsed.genderMix;
      if (parsed.atmosphere) $("atmosphere").value = parsed.atmosphere;
      renderChips(parsed);
    }

    function renderChips(parsed) {
      const entries = Object.entries(parsed);
      $("parsedChips").innerHTML = entries.length
        ? entries.map(([key, value]) => `<span class="chip">${key}: ${value}</span>`).join("")
        : `<span class="chip">인식된 조건 없음</span>`;
    }

    function scoreRestaurant(restaurant, scoreColumn, partySize, atmosphere, priority) {
      let score = Number(restaurant[scoreColumn] || 0);
      const reasons = [`모델 적합도 ${(score * 100).toFixed(1)}점`];

      const taste = Number(restaurant.taste_score || 0);
      const value = Number(restaurant.value_score || 0);
      const spaciousness = Number(restaurant.spaciousness_score || 0.5);
      const noise = Number(restaurant.noise_score || 0.5);

      if (taste >= 0.8) reasons.push("맛 점수 높음");
      if (value >= 0.7) reasons.push("가성비 점수 높음");
      if (restaurant.collected_review_count >= 30) reasons.push("리뷰 근거 충분");

      if (partySize >= 5) {
        const groupBoost = restaurant.seat_type.includes("group") ? 0.08 : 0;
        const spaciousBoost = spaciousness * 0.04;
        score += groupBoost + spaciousBoost;
        if (groupBoost) reasons.push("단체석 언급");
      } else if (partySize === 2 && restaurant.seat_type.includes("couple")) {
        score += 0.05;
        reasons.push("2인/데이트 좌석 언급");
      }

      if (atmosphere === "조용") {
        score += (1 - noise) * 0.05;
        reasons.push("조용한 분위기 반영");
      } else if (atmosphere === "활기") {
        score += noise * 0.05;
        reasons.push("활기 있는 분위기 반영");
      } else if (atmosphere === "넓은 곳") {
        score += spaciousness * 0.06;
        reasons.push("공간감 반영");
      }

      if (priority === "맛") {
        score += taste * 0.07;
        reasons.push("맛 우선 조건 반영");
      } else if (priority === "가성비") {
        score += value * 0.07;
        reasons.push("가성비 우선 조건 반영");
      } else if (priority === "분위기") {
        score += ((1 - noise) * 0.03) + (spaciousness * 0.03);
        reasons.push("분위기 우선 조건 반영");
      }

      return { score: Math.min(score, 1), reasons };
    }

    function recommend() {
      const area = $("area").value;
      const relation = $("relation").value;
      const occasion = $("occasion").value;
      const partySize = Number($("partySize").value || 1);
      const category = $("category").value;
      const priority = $("priority").value;
      const atmosphere = $("atmosphere").value;
      const label = `${relation}_${occasion}`;
      const scoreColumn = labelMap[label];

      const results = restaurants
        .filter((item) => item.area === area)
        .filter((item) => category === "상관없음" || item.category === category)
        .map((item) => ({ ...item, ...scoreRestaurant(item, scoreColumn, partySize, atmosphere, priority) }))
        .sort((a, b) => b.score - a.score)
        .slice(0, 10);

      $("summaryTitle").textContent = `${area} · ${relation} · ${occasion}`;
      $("summaryText").textContent = `${partySize}명, ${$("genderMix").value}, ${category}, ${atmosphere}, ${priority} 조건으로 정렬했어.`;

      if (results.length === 0) {
        $("resultList").innerHTML = `<div class="panel empty">조건에 맞는 추천 결과가 없어.</div>`;
        return;
      }

      $("resultList").innerHTML = results.map((item, index) => `
        <article class="item">
          <div class="item-head">
            <div>
              <div class="rank">#${index + 1}</div>
              <div class="name">${item.restaurant_name}</div>
              <div class="details">
                <span class="detail">${item.area}</span>
                <span class="detail">${item.category}</span>
                <span class="detail">별점 ${Number(item.rating || 0).toFixed(1)}</span>
                <span class="detail">리뷰 ${item.review_count}</span>
                <span class="detail">수집 ${item.collected_review_count}</span>
              </div>
            </div>
            <div class="score">${Math.round(item.score * 100)}</div>
          </div>
          <div class="details">
            ${item.reasons.map((reason) => `<span class="detail">${reason}</span>`).join("")}
          </div>
        </article>
      `).join("");
    }

    $("parseBtn").addEventListener("click", () => {
      applyParsed(parseNatural($("natural").value));
    });
    $("recommendBtn").addEventListener("click", recommend);
    $("natural").addEventListener("input", () => {
      applyParsed(parseNatural($("natural").value));
    });

    $("restaurantCount").textContent = restaurants.length.toLocaleString("ko-KR");
    applyParsed(parseNatural($("natural").value));
    recommend();
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path not in {"/", "/index.html"}:
            self.send_response(404)
            self.end_headers()
            return

        restaurants_json = json.dumps(load_restaurants(), ensure_ascii=False)
        page = HTML.replace("__RESTAURANTS__", restaurants_json)
        body = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


def main():
    host = "127.0.0.1"
    port = 8000
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"open http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
