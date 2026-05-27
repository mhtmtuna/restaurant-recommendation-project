# 장소 추천 ML 시스템

카카오맵 리뷰 기반으로 관계·상황·위치를 입력받아 적합한 음식점을 추천하는 멀티라벨 분류 시스템입니다.

---

## 바로 실행하기

```bash
git clone https://github.com/mhtmtuna/restaurant-recommendation-project.git
cd restaurant-recommendation-project
pip install -r requirements.txt
python src/web_app.py
```

브라우저에서 `http://127.0.0.1:8000` 접속. 자연어 입력과 선택형 입력을 모두 지원합니다.

GitHub에 `data/restaurants_features.csv`와 `data/restaurant_label_scores.csv`가 포함되어 있으므로, 단순 실행 시 크롤링과 모델 학습을 다시 할 필요가 없습니다.

---

## 터미널 추천 예시

```bash
python src/recommend.py --relation 연인 --occasion 식사 --area 강남
```

| 옵션 | 값 |
|------|----|
| `--relation` | `연인` / `친구` / `비즈니스` |
| `--occasion` | `식사` / `술자리` |
| `--area` | 강남, 건대, 잠실 등 |
| `--top-k` | 상위 N개 (기본 10) |

---

## 데이터 파이프라인

```
카카오맵 크롤링 (crawl_kakao.py)
    ↓
리뷰 → feature/라벨 변환 (build_features.py)
    ↓
멀티라벨 모델 학습 (train_model.py)
    ↓
추천 (recommend.py / web_app.py)
```

데이터를 새로 수집하거나 모델을 재학습할 때:

```bash
python src/crawl_kakao.py      # 중단 후 재실행 시 자동 이어서 진행
python src/build_features.py
python src/train_model.py
python src/web_app.py
```

---

## 주요 파일

| 파일 | 설명 |
|------|------|
| `src/web_app.py` | 웹 UI (자연어 + 선택형 입력) |
| `src/recommend.py` | 터미널 추천 실행 |
| `src/train_model.py` | 멀티라벨 모델 학습 (5-fold OOF) |
| `src/build_features.py` | 리뷰 → feature 및 라벨 생성 |
| `src/crawl_kakao.py` | 카카오맵 리뷰 수집 (Selenium) |
| `config/keywords.json` | feature / 라벨 키워드 사전 |
| `config/sampling_plan.json` | 크롤링 지역·카테고리·수집 수 설정 |
| `data/restaurants_features.csv` | 식당별 feature |
| `data/restaurant_label_scores.csv` | 식당별 추천 점수 (OOF) |
| `data/model_report.json` | 라벨별 성능 및 데이터 분포 |

---

## 핵심 설계

- 분위기를 **조도·소음·공간감·좌석 유형**으로 분해해 수치화
- 부정어 처리: "안 맛있어요" → 긍정 카운트 제외 (윈도우 12자 양방향 탐지)
- **Bayesian Shrinkage**: 리뷰가 적은 식당은 같은 지역×카테고리 평균으로 부드럽게 보정
- 한 식당이 여러 상황에 적합할 수 있으므로 **멀티라벨 분류** 사용
- 자연어 입력은 규칙 기반 슬롯 추출기로 구조화된 조건으로 변환

---

## 추천 점수 출처 구조

두 가지 점수 출처를 사용하는 하이브리드 구조입니다.

**기존 학습 식당 → `data/restaurant_label_scores.csv`**
- Out-of-Fold(OOF) 예측 점수 사용
- 각 식당은 자신이 학습에 포함되지 않은 fold의 모델로 예측 → 데이터 누수 없음

**신규 식당 → `models/restaurant_recommender.joblib`**
- 전체 데이터로 학습된 최종 모델이 실시간 추론
- `--score-source csv|model|auto` 옵션으로 선택 가능

---

## 현재 한계

| 항목 | 현재 상태 | 개선 방향 |
|------|---------|---------|
| Positive sample 부족 | 807개 식당 수집 후에도 couple 29개·business_meal 17개·business_drink 13개 — 권장(30개) 미달. 간접 키워드 확장 및 피처 기반 보완 규칙을 시도했으나 개선 폭 미미 | 카카오맵 리뷰는 관계 맥락을 잘 명시하지 않아 수집량을 늘려도 비율이 크게 개선되기 어려움. LLM 기반 라벨링 또는 human-labeled 검증셋 확보가 근본 해결책 |
| 라벨 생성 방식 | 키워드 사전 기반 자동 라벨링 → 리뷰에 "여친이랑", "회식" 등 명시적 표현이 없으면 라벨 미할당 | human-labeled 검증셋 확보 또는 LLM 기반 라벨링 |
| 텍스트 표현 | 키워드 빈도 기반 (TF 수준) | 임베딩(KLUE-BERT 등) 도입으로 문맥 이해 |
| 지역 커버리지 | 강남·건대·잠실 3개 지역 | 서울 전역 확장 |
| 가격 데이터 | 카카오맵에서 수집 불가 — 전체 NULL | 다른 소스(네이버 플레이스 등) 보완 수집 |

---

## 주의

카카오맵 화면 구조가 바뀌면 CSS 선택자를 조정해야 할 수 있습니다. 수집은 공개 페이지를 대상으로 천천히 진행하고, 서비스 약관과 robots 정책을 확인한 뒤 사용하세요.
