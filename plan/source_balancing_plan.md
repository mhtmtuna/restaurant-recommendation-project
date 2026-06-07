# 카카오·네이버 데이터 통합 개선 계획

## 목적

카카오와 네이버 리뷰를 함께 사용하되, 모델이 음식점 특성보다 수집 출처 차이를 학습하는 문제를 줄인다.

## 현재 확인된 문제

- 카카오 식당은 347개, 네이버 식당은 283개다.
- 식당당 수집 리뷰 수 평균은 카카오 34.7개, 네이버 10.6개다.
- 관계 라벨 positive 비율 차이가 크다.
  - `couple`: 카카오 2.9%, 네이버 50.9%
  - `friend_meal`: 카카오 3.2%, 네이버 54.4%
  - `friend_drink`: 카카오 3.2%, 네이버 59.0%
- source-aware 평가에서 교차 출처 일반화 성능이 낮다.
  - 카카오 학습 → 네이버 검증: macro F1 0.0000
  - 네이버 학습 → 카카오 검증: macro F1 0.1369

## 다음 작업 순서

1. 식당당 리뷰를 출처별 최대 10개로 균형 샘플링한다.
2. `식당명 + 지역 + 카테고리` 정규화 기준으로 카카오·네이버 식당 ID 매칭 후보 CSV를 만든다.
3. 자동 매칭 신뢰도가 낮은 식당은 별도 CSV로 분리해 수동 확인한다.
4. 출처별 메타데이터를 분리한다.
   - `kakao_rating`, `naver_rating`
   - `kakao_review_count`, `naver_review_count`
5. 출처별 리뷰 feature와 통합 feature 사용 여부를 비교한다.
6. 다음 세 가지 실험을 반복한다.
   - 모든 feature 사용
   - `collected_review_count` 제외
   - `rating`, `review_count`, `collected_review_count` 제외
7. 혼합 OOF와 source-aware 평가를 함께 보고 최종 구성을 선택한다.

## 판단 기준

- 혼합 OOF 성능만으로 모델을 선택하지 않는다.
- source-aware 성능이 함께 개선되는 구성을 우선한다.
- 수집량 메타데이터가 출처 proxy로 작동하는지 ablation으로 확인한다.
- feature 제거는 p-value만으로 결정하지 않고 Random Forest ablation 결과를 함께 본다.

