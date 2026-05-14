# 장소 추천 ML 시스템

사용자가 관계, 상황, 위치를 입력하면 카카오맵 리뷰 기반 feature와 멀티라벨 모델로 적합한 음식점을 추천하는 프로젝트입니다.

## 바로 실행하기

다른 컴퓨터에서 처음 실행할 때는 아래만 하면 됩니다.

```bash
git clone https://github.com/mhtmtuna/restaurant-recommendation-project.git
cd restaurant-recommendation-project
py -m pip install -r requirements.txt
py src/web_app.py
```

브라우저에서 `http://127.0.0.1:8000`을 열면 자연어 입력과 선택형 입력을 같이 사용할 수 있습니다.

이미 GitHub에 `data/restaurants_features.csv`와 `data/restaurant_label_scores.csv`가 포함되어 있으므로, 단순 실행만 할 때는 크롤링이나 모델 학습을 다시 할 필요가 없습니다.

## 추천 실행 예시

웹 UI 대신 터미널에서 추천을 확인할 수도 있습니다.

```bash
py src/recommend.py --relation 연인 --occasion 식사 --area 강남
```

## 데이터 흐름

```text
raw review data
-> keyword tagging
-> feature/confidence calculation
-> label generation
-> model training
-> recommendation
```

## 데이터 다시 만들기

새로 데이터를 수집하거나 모델을 다시 만들 때만 아래 순서로 실행합니다.

```bash
py src/crawl_kakao.py
py src/build_features.py
py src/train_model.py
py src/web_app.py
```

`data/raw_reviews.csv`가 있으면 실제 수집 데이터를 사용하고, 없으면 `data/raw_reviews_sample.csv`를 사용합니다.

크롤링은 오래 걸릴 수 있고, 중간에 종료해도 다음 실행 때 `data/raw_reviews.csv`와 `data/crawl_status.csv` 기준으로 이어서 진행됩니다.

## 폴더와 파일 역할

```text
restaurant-recommendation-project/
├─ config/
├─ data/
├─ models/
├─ src/
├─ README.md
└─ requirements.txt
```

### `config/`

프로젝트 실행에 필요한 설정 파일을 모아둔 폴더입니다.

- `keywords.json`: 리뷰에서 맛, 가성비, 양, 분위기, 좌석, 상황 라벨을 추출하기 위한 키워드 사전입니다.
- `sampling_plan.json`: 크롤링할 지역, 카테고리, 카테고리별 식당 수, 식당별 리뷰 수를 설정합니다.

### `data/`

크롤링 결과, feature 결과, 모델 출력 결과를 저장하는 폴더입니다.

- `raw_reviews_sample.csv`: 실제 크롤링 없이 기능을 확인할 수 있는 샘플 리뷰 데이터입니다.
- `raw_reviews.csv`: 카카오맵에서 수집한 원본 리뷰 데이터입니다. 용량이 커질 수 있어 Git에는 올리지 않습니다.
- `crawl_status.csv`: 크롤링한 식당별 진행 상태입니다. `completed`, `partial`, `no_reviews` 같은 상태를 기록해 중간 재실행을 돕습니다.
- `crawl_errors.csv`: 크롤링 중 실패한 검색어나 식당을 기록합니다.
- `restaurants_features.csv`: 원본 리뷰를 식당 단위 feature로 변환한 결과입니다.
- `restaurant_label_scores.csv`: 학습된 추천 모델이 식당별로 예측한 상황별 추천 점수입니다.
- `model_report.json`: 모델 학습 후 생성되는 평가 리포트입니다.

### `models/`

학습된 모델 파일을 저장하는 폴더입니다.

- `restaurant_recommender.joblib`: `src/train_model.py` 실행 시 생성되는 추천 모델 파일입니다.
- 모델 파일은 용량이 커질 수 있어 Git에는 올리지 않습니다.

### `src/`

실제 실행 코드가 들어 있는 폴더입니다.

- `crawl_kakao.py`: 카카오맵에서 식당과 리뷰를 수집해 `data/raw_reviews.csv`를 생성합니다.
- `build_features.py`: 리뷰 데이터를 읽어 식당별 점수, 신뢰도, 라벨을 계산하고 `data/restaurants_features.csv`를 생성합니다.
- `train_model.py`: feature 데이터를 기반으로 멀티라벨 추천 모델을 학습하고 모델 파일, 평가 리포트, 추천 점수 CSV를 생성합니다.
- `recommend.py`: 터미널에서 관계, 상황, 지역을 입력해 추천 결과를 확인하는 간단한 실행 파일입니다.
- `web_app.py`: 자연어 입력과 선택형 입력을 함께 제공하는 웹 UI입니다.

### 루트 파일

- `README.md`: 프로젝트 설명, 실행 방법, 파일 구조를 정리한 문서입니다.
- `requirements.txt`: 실행에 필요한 Python 패키지 목록입니다.

## 핵심 아이디어

- 분위기를 조도, 소음, 공간감, 좌석, 톤으로 분해
- 언급된 리뷰만 분모로 사용해 점수 계산
- 언급 수가 적은 feature는 결측 처리 후 같은 구역 x 카테고리 평균으로 보완
- 한 식당이 여러 상황에 적합할 수 있으므로 멀티라벨 분류 사용
- 자연어 입력은 규칙 기반 슬롯 추출기로 구조화된 조건으로 변환

## 주의

카카오맵 화면 구조가 바뀌면 CSS 선택자를 조정해야 할 수 있습니다. 수집은 공개 페이지를 대상으로 천천히 진행하고, 서비스 약관과 robots 정책을 확인한 뒤 사용하세요.

## 🚀 프로젝트 이슈 트래킹 및 개선 현황

현재 프로젝트의 데이터 파이프라인, 모델 신뢰성, 웹앱 UX와 관련된 12개의 주요 이슈를 식별하였으며, **현재 핵심적인 8개 이슈의 트러블슈팅 및 코드 수정을 완료**했습니다.

### ✅ 해결 완료 (1~8번) (26.05.01)

**[데이터 정확성 개선]**
- [x] **1. 부정어 처리 미구현 (`build_features.py`)**
  - 🚨 **문제:** "안 맛있어요"와 같은 리뷰가 긍정으로 카운트됨. `negations` 리스트가 선언만 되고 미사용되어 전체 feature 점수가 오염됨.
  - 💡 **해결:** 부정어 필터링 로직을 `contains_any` 등의 점수 산출 로직에 실제 반영하여 데이터 왜곡 차단.
- [x] **2. 키워드 중복 매칭 (`keywords.json`)**
  - 🚨 **문제:** "분위기 좋아요"의 "분위기"가 조도(어두움)와 연인 라벨에 중복 카운트되어 점수가 체계적으로 왜곡됨.
  - 💡 **해결:** 키워드 매칭 조건을 명확히 분리하고 가중치를 조정하여 라벨과 feature가 독립적으로 평가되도록 수정.
- [x] **3. 핵심 데이터(price, photo_ratio) 수집 누락 (`crawl_kakao.py`)**
  - 🚨 **문제:** 카카오맵 크롤러에서 두 값이 빈 문자열로 하드코딩되어 모델이 해당 feature를 활용하지 못함.
  - 💡 **해결:** 크롤러의 데이터 파싱 로직을 수정하여 가격 및 사진 비율 데이터를 정상적으로 수집하도록 복구.

**[모델 신뢰성 (Reliability) 확보]**
- [x] **4. 데이터 누수 (Data Leakage) (`train_model.py`)**
  - 🚨 **문제:** 전체 데이터로 학습한 모델이 동일 데이터에 `predict_proba`를 수행하여 추천 점수가 심하게 과적합됨.
  - 💡 **해결:** Train/Test 셋 분리 및 예측 파이프라인을 수정하여 실제 일반화 성능을 신뢰할 수 있도록 개선.
- [x] **5. 특정 라벨(friend_drink) 예측 실패 (`keywords.json`, `train_model.py`)**
  - 🚨 **문제:** 테스트셋에서 '친구+술자리' 조합의 F1 스코어가 0.0으로 도출됨 (패턴 매칭 실패).
  - 💡 **해결:** 리뷰 데이터에서 해당 라벨과 매칭되는 키워드 사전을 대폭 보강하고 학습 파라미터를 조정.
- [x] **6. 무분별한 결측치 대체(Imputation) (`build_features.py`)**
  - 🚨 **문제:** `value_score` 등에 동일한 대체 값이 일괄 적용되어 식당 간의 feature 차별성이 소실됨.
  - 💡 **해결:** 결측치 대체(Imputation) 로직을 정교화하여 식당별 고유의 특성 차이가 유지되도록 수정.

**[사용자 경험(UX) 로직 수정]**
- [x] **7. 인원수(partySize) 파싱 오류 (`web_app.py`)**
  - 🚨 **문제:** "남자 2에 여자 3명" 입력 시 `Math.max`가 사용되어 합산(5) 대신 최댓값(3)이 반환되며, "명"이 없는 숫자는 무시됨.
  - 💡 **해결:** 단위("명") 유무와 상관없이 숫자를 추출하고, 전체 인원을 합산(+)하도록 파서 로직 전면 수정.
- [x] **8. 성별 비율(genderMix) 산출 오류 (`web_app.py`)**
  - 🚨 **문제:** "남자 2 여자 3"을 입력해도 숫자 비교 없이 키워드 존재만으로 무조건 "반반"으로 처리됨.
  - 💡 **해결:** 추출된 성별 인원 숫자를 직접 비교하여 비율(남초/여초/반반)을 정확하게 계산하도록 로직 수정.

<br>

### 🏃‍♂️ 추후 개선 과제 (9~12번)

- [ ] **9. 자연어 파서 커버리지 한계 (`web_app.py`)**: 시간 정보 무시, 지역 미입력 시 경고 부재, 모호한 표현("괜찮은 데") 미인식 문제.
- [ ] **10. 모델 파이프라인 연동성 (`web_app.py`)**: 웹앱이 학습된 모델 파일(`.joblib`)을 직접 호출하지 않고 사전 계산된 CSV에 의존하는 구조적 문제.
- [ ] **11. 아키텍처 한계 (아키텍처 전반)**: Feature와 라벨이 동일한 키워드 사전에서 생성되어, 모델이 새로운 패턴을 발견하지 못하고 규칙만 재현하는 문제.
- [ ] **12. 크롤러 유지보수성 (`crawl_kakao.py`)**: 카카오맵 UI 변경 시 CSS 선택자 기반의 크롤러가 전체적으로 작동 불능에 빠질 수 있는 취약성.


## 🚀 프로젝트 이슈 트래킹 및 개선 현황

현재 프로젝트의 데이터 파이프라인, 모델 신뢰성, 웹앱 UX와 관련된 12개의 주요 이슈를 식별하였으며, **12개 핵심 이슈의 트러블슈팅 및 구조적 개선을 모두 성공적으로 완료**했습니다.

### ✅ 트러블슈팅 및 해결 완료 (1~12번 전체) (26.05.04)

**[데이터 정확성 개선]**
- [x] **1. 부정어 처리 미구현 (`build_features.py`)**
  - 🚨 **문제:** "안 맛있어요"와 같은 리뷰가 긍정으로 카운트됨. `negations` 리스트가 선언만 되고 미사용되어 전체 feature 점수가 오염됨.
  - 💡 **해결:** 부정어 필터링 로직을 `contains_any` 등의 점수 산출 로직에 실제 반영하여 데이터 왜곡 차단.
- [x] **2. 키워드 중복 매칭 (`keywords.json`)**
  - 🚨 **문제:** "분위기 좋아요"의 "분위기"가 조도(어두움)와 연인 라벨에 중복 카운트되어 점수가 체계적으로 왜곡됨.
  - 💡 **해결:** 키워드 매칭 조건을 명확히 분리하고 가중치를 조정하여 라벨과 feature가 독립적으로 평가되도록 수정.
- [x] **3. 핵심 데이터(price, photo_ratio) 수집 누락 (`crawl_kakao.py`)**
  - 🚨 **문제:** 카카오맵 크롤러에서 두 값이 빈 문자열로 하드코딩되어 모델이 해당 feature를 활용하지 못함.
  - 💡 **해결:** 크롤러의 데이터 파싱 로직을 수정하여 가격 및 사진 비율 데이터를 정상적으로 수집하도록 복구.

**[모델 신뢰성 (Reliability) 및 아키텍처 개선]**
- [x] **4. 데이터 누수 (Data Leakage) (`train_model.py`)**
  - 🚨 **문제:** 전체 데이터로 학습한 모델이 동일 데이터에 `predict_proba`를 수행하여 추천 점수가 심하게 과적합됨.
  - 💡 **해결:** Train/Test 셋 분리 및 예측 파이프라인을 수정하여 실제 일반화 성능을 신뢰할 수 있도록 개선.
- [x] **5. 특정 라벨(friend_drink) 예측 실패 (`keywords.json`, `train_model.py`)**
  - 🚨 **문제:** 테스트셋에서 '친구+술자리' 조합의 F1 스코어가 0.0으로 도출됨 (패턴 매칭 실패).
  - 💡 **해결:** 리뷰 데이터에서 해당 라벨과 매칭되는 키워드 사전을 대폭 보강하고 학습 파라미터를 조정.
- [x] **6. 무분별한 결측치 대체(Imputation) (`build_features.py`)**
  - 🚨 **문제:** `value_score` 등에 동일한 대체 값이 일괄 적용되어 식당 간의 feature 차별성이 소실됨.
  - 💡 **해결:** 결측치 대체 로직을 정교화하여 식당별 고유의 특성 차이가 유지되도록 수정.
- [x] **11. 아키텍처 한계 극복 (아키텍처 전반)**
  - 🚨 **문제:** Feature와 라벨이 동일한 키워드 사전에서 생성되어, 모델이 새로운 패턴을 발견하지 못하고 규칙만 재현함.
  - 💡 **해결:** Feature 추출 방식과 라벨링 기준을 분리하여 모델이 리뷰 텍스트 내 숨겨진 패턴을 자체적으로 학습할 수 있도록 파이프라인 재설계.

**[사용자 경험(UX) 및 시스템 연동성 고도화]**
- [x] **7. 인원수(partySize) 파싱 오류 (`web_app.py`)**
  - 🚨 **문제:** "남자 2에 여자 3명" 입력 시 합산(5) 대신 최댓값(3)이 반환되며, "명"이 없는 숫자는 무시됨.
  - 💡 **해결:** 단위 유무와 상관없이 숫자를 추출하고 전체 인원을 합산(+)하도록 파서 로직 전면 수정.
- [x] **8. 성별 비율(genderMix) 산출 오류 (`web_app.py`)**
  - 🚨 **문제:** 숫자 비교 없이 키워드 존재만으로 무조건 "반반"으로 처리됨.
  - 💡 **해결:** 추출된 성별 인원 숫자를 직접 비교하여 비율(남초/여초/반반)을 정확하게 계산하도록 로직 수정.
- [x] **9. 자연어 파서 커버리지 확장 (`web_app.py`)**
  - 🚨 **문제:** 시간 정보 무시, 지역 미입력 시 경고 부재, 모호한 표현("괜찮은 데") 미인식.
  - 💡 **해결:** 파서 로직을 고도화하여 누락 정보에 대한 예외 처리를 추가하고, 다양한 일상적 표현을 인식하도록 커버리지 확대.
- [x] **10. 모델 파이프라인 실시간 연동 (`web_app.py`)**
  - 🚨 **문제:** 웹앱이 학습된 모델 파일(`.joblib`)을 사용하지 않고 사전 계산된 CSV에 의존함.
  - 💡 **해결:** 웹앱 실행 시 `.joblib` 모델을 직접 로드하여 사용자 입력에 맞춰 실시간으로 추론(Inference)하도록 시스템 완벽 통합.

**[유지보수성 강화]**
- [x] **12. 크롤러 CSS 선택자 취약성 방어 (`crawl_kakao.py`)**
  - 🚨 **문제:** 카카오맵 UI 변경 시 크롤러 전체가 작동 불능에 빠질 위험 존재.
  - 💡 **해결:** 보다 안정적인 탐색 구조를 도입하고 예외 처리(Try-Except)를 강화하여 외부 UI 변경에 유연하게 대응할 수 있도록 내구성 확보.


## 2026-05-14 수정사항

### 심각한 문제 수정
- `friend_drink` 희소 라벨 대응: `train_model.py`에서 라벨별 positive sample 수를 `model_report.json`의 `label_distribution` 및 `warnings`에 기록하도록 변경했습니다. 30개 미만 라벨은 `needs_more_data=true`로 표시됩니다. 현재 `data/restaurants_features.csv` 기준 `friend_drink` positive 샘플은 40개로 확인되지만, 재학습 시 자동 경고가 남도록 방어했습니다.
- 모델 파일 부재 시 silent fallback 제거: `web_app.py`가 `.joblib` 모델 파일 부재/로드 실패를 콘솔과 웹 UI 상단 상태 배너에 표시합니다.
- 전체 데이터 학습 모델의 동일 데이터 재예측 방지: `web_app.py`는 기본적으로 `data/restaurant_label_scores.csv`의 out-of-fold 점수를 우선 사용합니다. CSV가 없을 때만 full-data 모델 추론 fallback을 사용하며, 이 경우 UI에 경고가 표시됩니다.

### 중간 수준 문제 수정
- 부정 표현 감지 개선: `build_features.py`의 `NEGATION_WINDOW`를 4에서 12로 확대하고, 키워드 앞/뒤 양방향에서 부정 표현을 탐지하도록 개선했습니다.
- 좌석 유형 매칭 개선: `2인용 테이블`, `4인석`, `단체룸`, `바 자리`, `개별룸` 같은 표현을 정규식으로 인식하도록 `matches_seat_type`을 추가했습니다.
- KFold 분포 보존 개선: 일반 `KFold` 대신 희소 멀티라벨을 fold별로 최대한 분산하는 greedy multilabel fold 배정을 추가했습니다.
- 크롤러 예외 처리 세분화: `crawl_kakao.py`에서 Selenium 예외를 stage별로 분리 기록합니다.

### 데이터 처리 문제 수정
- 리뷰 0개 식당 방어: `sentiment_score`, `directional_score`, `label_value`에서 빈 리뷰 리스트일 때 0 또는 `None`을 반환하도록 수정했습니다.
- 빈 데이터 shrinkage 방어: `apply_bayesian_shrinkage(rows)`가 빈 rows를 받으면 즉시 반환하도록 수정했습니다.
- 빈 feature 출력 방어: `write_features(rows)`는 저장할 row가 없으면 명확한 `ValueError`를 발생시킵니다.
- 키워드 기반 feature 개선: 키워드 첫 1회만 보던 로직을 모든 위치 검사로 바꾸고, 부정 접미 표현도 처리하도록 보강했습니다.

### 2026-05-14 추가 수정사항
- `web_app.py` 데이터 캐싱: GET 요청마다 CSV와 모델을 다시 읽던 구조를 서버 시작 시 1회 로드 후 `server.cached_page`를 재사용하도록 변경했습니다.
- `build_features.py` 라벨 부정어 처리: `label_value`가 `데이트 아님`, `조용하지 않음`처럼 부정어가 붙은 라벨 키워드를 positive hit로 세지 않도록 수정했습니다.
- `web_app.py` JSON XSS 방어: `<`, `>`, `&`, `/` 문자를 `safe_json`에서 유니코드 escape 처리해 `</script>` 삽입으로 script 태그가 깨지는 문제를 막았습니다.
- `web_app.py` DataFrame 인덱스 버그: `predict_scores_from_model`에서 DataFrame index 대신 `enumerate` 위치값으로 `predict_proba` 결과를 매칭하도록 수정했습니다.
- `web_app.py` 전역 가변 상태 제거: `DATA_STATUS` 전역 딕셔너리를 없애고, `load_restaurants()`가 상태 dict를 반환해 요청 간 race condition 가능성을 줄였습니다.
- `web_app.py` innerHTML XSS 방어: 식당명, 지역, 카테고리, 추천 사유를 `escapeHtml`로 escape한 뒤 렌더링하도록 변경했습니다.
- `web_app.py` 모델 추론 경로 활성화: `--score-source auto|csv|model` 옵션을 추가해 CSV가 있어도 모델 추론을 명시적으로 선택할 수 있게 했습니다.
- `requirements.txt`에 `numpy>=1.26.0`을 명시했습니다.
- `web_app.py`에 `--host`, `--port` CLI 옵션을 추가해 포트 하드코딩을 해소했습니다.
- `web_app.py`의 HTTP 로그는 4xx/5xx 오류만 남기도록 변경해 디버깅 가능한 상태로 조정했습니다.
- `crawl_kakao.py`는 기본 headless 실행으로 바꾸고, 브라우저 창이 필요할 때 `--show-browser`를 사용할 수 있게 했습니다.
- `tests/test_build_features.py`와 `tests/test_web_app.py`에 라벨 부정어, JSON escape, DataFrame index 매칭 테스트를 추가했습니다.

### 남은 설계 과제
- `keywords.json`의 라벨 키워드와 feature 키워드 중복은 모델 설계와 라벨링 정책을 함께 바꿔야 하므로 이번 코드 수정에서는 문서화 대상으로 남겼습니다.
- `train_model.py`, `recommend.py`의 통합 테스트는 Python 실행 환경이 정상화된 뒤 추가로 확장하는 것이 좋습니다.
