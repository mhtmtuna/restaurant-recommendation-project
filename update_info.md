# Keywords Update Report

## 변경 대상

- `config/keywords.json`

## 수정 원칙

1. `features` 키워드는 속성어만 유지
   - taste
   - value
   - portion
   - brightness
   - noise
   - spaciousness
   - seat_type
2. `labels` 키워드는 상황어만 유지
   - 관계
   - 목적
   - 행위
3. `features`와 `labels` 사이에 키워드가 겹치지 않도록 정리

## 주요 수정 내용

### features 정리

`seat_type` feature에 들어 있던 상황어를 제거했습니다.

- `seat_type.couple`
  - 제거: `데이트`
  - 유지: `2인석`, `창가`
- `seat_type.group`
  - 제거: `회식`, `모임`
  - 유지: `단체석`, `테이블 붙`

### labels 정리

#### `labels.business_meal`

제거한 속성어/기능어:

- `조용`
- `예약`
- `룸`

변경 후 키워드:

- `회식`
- `미팅`
- `거래처`
- `상견례`
- `비즈니스 식사`
- `접대 식사`
- `회사 점심`
- `팀 점심`

#### `labels.business_drink`

제거한 속성어/기능어:

- `넓`
- `룸`
- `안주`

변경 후 키워드:

- `회식 자리`
- `거래처 술자리`
- `접대`
- `2차 회식`
- `회사 술자리`
- `팀 회식`
- `비즈니스 술자리`

#### `labels.couple_meal`

제거한 속성어:

- `분위기 좋`
- `감성`
- `무드`

변경 후 키워드:

- `데이트`
- `여친`
- `남친`
- `기념일`
- `소개팅`
- `커플`
- `연인`

#### `labels.couple_drink`

제거한 속성어/장소어:

- `분위기 좋`
- `무드`
- `와인`
- `칵테일`
- `바`

변경 후 키워드:

- `데이트`
- `여친`
- `남친`
- `기념일`
- `소개팅`
- `커플`
- `연인`

#### `labels.friend_drink`

제거한 feature 중복 키워드:

- `시끌`

변경 후 키워드:

- `친구`
- `친구랑`
- `친구들`
- `동기`
- `한잔`
- `2차`
- `술자리`
- `뒤풀이`
- `불금`
- `퇴근 후 한잔`
- `친구 술자리`
- `같이 마시`
- `다같이 마시`

## 키워드 중복 검증

`features` 전체 키워드 set과 `labels` 전체 키워드 set을 추출해서 교집합을 확인했습니다.

```text
INTERSECTION
[]
intersection_empty= True
```

검증 결과, `features`와 `labels` 사이의 직접 중복 키워드는 없습니다.

## 실행 결과

### `build_features.py`

실행 명령:

```powershell
.\.venv\Scripts\python.exe src\build_features.py
```

결과:

```text
saved: C:\Users\frado\가천대\인공지능프로그래밍\팀플\AI_TeamProject_recommendation\data\restaurants_features.csv
```

주의:

현재 workspace에 `data/raw_reviews.csv`가 없어서 `src/build_features.py`가 `data/raw_reviews_sample.csv`로 fallback했습니다. 그 결과 `data/restaurants_features.csv`가 샘플 데이터 기준 2개 식당으로 다시 생성되었습니다.

### `train_model.py`

실행 명령:

```powershell
.\.venv\Scripts\python.exe src\train_model.py
```

결과:

```text
ValueError: Need at least 10 restaurants to train a baseline model.
```

원인:

- 재학습에 필요한 원본 리뷰 파일 `data/raw_reviews.csv`가 현재 workspace에 없습니다.
- `.gitignore`에 `data/raw_reviews.csv`가 포함되어 있어 Git 추적 대상도 아닙니다.
- sample fallback으로 생성된 feature 데이터가 2개 식당뿐이라 `train_model.py`의 최소 학습 조건인 10개 식당을 만족하지 못했습니다.

## Macro F1 비교

요청한 비교 기준:

- 이전 결과: `0.51`

이번 실행 결과:

- 새 `model_report.json`은 생성되지 않았습니다.
- 따라서 새 라벨 기준 macro F1은 산출하지 못했습니다.

참고:

현재 남아 있는 기존 `data/model_report.json`의 macro F1은 `0.5909621327429546`이지만, 이번에 수정한 새 라벨 기준 재학습 결과는 아닙니다.

## 남은 작업

새 라벨 기준으로 모델을 재학습하려면 `data/raw_reviews.csv` 원본 리뷰 데이터가 필요합니다. 해당 파일을 `data/` 폴더에 둔 뒤 아래 순서로 다시 실행하면 됩니다.

```powershell
.\.venv\Scripts\python.exe src\build_features.py
.\.venv\Scripts\python.exe src\train_model.py
```

