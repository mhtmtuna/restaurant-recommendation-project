# 장소 추천 ML 시스템

사용자가 관계, 상황, 위치를 입력하면 카카오맵 리뷰 기반 feature와 멀티라벨 모델로 적합한 음식점을 추천하는 프로젝트입니다.

## 먼저 하는 순서

1. `config/keywords.json` 키워드 사전 보강
2. `config/sampling_plan.json`에서 수집 범위 확인
3. `src/crawl_kakao.py`로 리뷰 데이터 수집
4. `src/build_features.py`로 식당별 feature 생성
5. 모델 학습 및 추천 인터페이스 구현

## 데이터 흐름

raw review data -> keyword tagging -> feature/confidence calculation -> label generation -> model training -> recommendation

## 실행

필요 패키지 설치:

```bash
pip install -r requirements.txt
```

카카오맵 리뷰 수집:

```bash
python src/crawl_kakao.py
```

feature 생성:

```bash
python src/build_features.py
```

모델 학습:

```bash
python src/train_model.py
```

추천 실행 예시:

```bash
python src/recommend.py --relation 연인 --occasion 식사 --area 강남
```

`data/raw_reviews.csv`가 있으면 실제 수집 데이터를 사용하고, 없으면 `data/raw_reviews_sample.csv`를 사용합니다.

## 핵심 아이디어

- 분위기를 조도, 소음, 공간감, 좌석, 톤으로 분해
- 언급된 리뷰만 분모로 사용해 점수 계산
- 언급 수가 적은 feature는 결측 처리 후 같은 구역 x 카테고리 평균으로 보완
- 한 식당이 여러 상황에 적합할 수 있으므로 멀티라벨 분류 사용

## 주의

카카오맵 화면 구조가 바뀌면 CSS 선택자를 조정해야 할 수 있습니다. 수집은 공개 페이지를 대상으로 천천히 진행하고, 서비스 약관과 robots 정책을 확인한 뒤 사용하세요.
