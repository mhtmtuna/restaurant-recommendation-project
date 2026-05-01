# 장소 추천 ML 시스템

사용자가 관계, 상황, 위치를 입력하면 카카오맵 리뷰 기반 feature와 멀티라벨 모델로 적합한 음식점을 추천하는 프로젝트입니다.

## 먼저 하는 순서

1. `config/keywords.json` 키워드 사전 보강
2. `data/raw_reviews_sample.csv` 형식에 맞춰 리뷰 데이터 수집
3. `src/build_features.py`로 식당별 feature 생성
4. 모델 학습 및 추천 인터페이스 구현

## 데이터 흐름

raw review data -> keyword tagging -> feature/confidence calculation -> label generation -> model training -> recommendation

## 핵심 아이디어

- 분위기를 조도, 소음, 공간감, 좌석, 톤으로 분해
- 언급된 리뷰만 분모로 사용해 점수 계산
- 언급 수가 적은 feature는 결측 처리 후 같은 구역 x 카테고리 평균으로 보완
- 한 식당이 여러 상황에 적합할 수 있으므로 멀티라벨 분류 사용
