# 모델 선택 및 Feature 분석 요약

## 데이터 기준

- 카카오 리뷰: 12,044개
- 네이버 리뷰: 3,001개
- 병합 리뷰: 15,045개
- 식당 feature: 630개
- 평가 방식: 5-fold out-of-fold multilabel 평가

## 모델 선택

| 모델 | R² | Macro F1 | Micro F1 | 판단 |
|---|---:|---:|---:|---|
| Random Forest | 0.3716 | 0.6201 | 0.6688 | 선택 |
| CatBoost | 0.3362 | 0.5422 | 0.5996 | RF보다 낮음 |
| AdaBoost | -0.1230 | 0.5891 | 0.6245 | 확률 품질이 낮음 |

Random Forest를 선택한다. 세 모델 중 R², macro F1, micro F1이 모두 가장 높다.

## 제거 완료 Feature

- `price`
- `photo_ratio`

두 컬럼은 전체 행이 결측이므로 모델 입력에서 제거했다. UI 표시용 컬럼은 유지한다.

## 핵심 Feature

| Feature | RF importance | 유의 라벨 수 | 최소 보정 p-value | 제거 시 ΔR² | 제거 시 ΔMacro F1 |
|---|---:|---:|---:|---:|---:|
| `taste_confidence` | 0.1602 | 5 | 1.68e-49 | - | - |
| `collected_review_count` | 0.0967 | 5 | 1.63e-60 | -0.0112 | -0.0367 |
| `spaciousness_confidence` | 0.0574 | 5 | 1.41e-12 | - | - |
| `review_count` | 0.0563 | 5 | 6.64e-11 | -0.0033 | -0.0075 |
| `rating` | 0.0461 | 5 | 1.59e-13 | +0.0025 | -0.0115 |

`collected_review_count`, `review_count`, `rating`은 성능에 기여하지만 출처 차이를 간접적으로 나타내는 proxy일 가능성이 있다. 다음 데이터 균형 실험에서 다시 검증한다.

## 제거 검토 Feature

| Feature | RF importance | 최소 보정 p-value | 제거 시 ΔR² | 제거 시 ΔMacro F1 | 판단 |
|---|---:|---:|---:|---:|---|
| `taste_score` | 0.0341 | 8.02e-02 | -0.0009 | +0.0133 | 제거 후보 |
| `value_confidence` | 0.0230 | 8.09e-02 | -0.0011 | +0.0036 | 추가 검증 |
| `portion_mentions` | 0.0098 | 1.14e-01 | -0.0017 | +0.0047 | 추가 검증 |
| `brightness_confidence` | 0.0019 | 9.22e-02 | -0.0035 | -0.0205 | 유지 |
| `brightness_mentions` | 0.0019 | 2.43e-01 | -0.0046 | -0.0199 | 유지 |

p-value만으로 feature를 제거하지 않는다. `brightness_confidence`, `brightness_mentions`는 통계적 유의성이 약하지만 ablation에서 제거 시 성능이 떨어진다.

## R² 해석

- 모델별 `R²`: 각 모델의 OOF 확률 점수가 실제 라벨을 얼마나 설명하는지 나타낸다.
- feature별 `ΔR²`: 해당 feature를 제거한 뒤 모델 전체 R²가 얼마나 변하는지 나타낸다.
- feature 자체의 단독 R²보다 `ΔR²`가 실제 제거 판단에 더 직접적이다.

## 출처 편향 주의

| 학습 | 검증 | R² | Macro F1 |
|---|---|---:|---:|
| 카카오+네이버 혼합 OOF | 혼합 | 0.3716 | 0.6201 |
| 카카오만 | 네이버만 | -0.3702 | 0.0000 |
| 네이버만 | 카카오만 | -2.9535 | 0.1369 |

혼합 OOF 점수를 일반화 성능으로 그대로 주장하지 않는다. 다음 단계는 `source_balancing_plan.md`를 따른다.

## 산출물

- `experiments/model_feature_analysis_dashboard.png`
- `experiments/model_metrics_summary.csv`
- `experiments/feature_analysis_summary.csv`
- `experiments/feature_significance.csv`
- `experiments/feature_ablation.csv`
- `experiments/source_aware_evaluation.csv`

