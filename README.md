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
