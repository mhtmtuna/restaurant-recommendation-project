# -*- coding: utf-8 -*-
"""README/분석 리포트용 차트 생성 (미니멀 통일 스타일).

레포에 커밋된 실험 산출물만 입력으로 사용한다 (스크린샷 X, 100% 재현 가능):
  - experiments/model_metrics_summary.csv      → 모델 비교
  - experiments/feature_importance.json        → 피처 중요도
  - experiments/feature_ablation.csv           → 피처 ablation
  - experiments/source_aware_evaluation.json   → 출처 편향

실행:  python experiments/make_readme_figures.py
출력:  docs/fig_*.png
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
OUT = os.path.join(ROOT, "docs")
os.makedirs(OUT, exist_ok=True)

# 한글 폰트
for cand in ["Malgun Gothic", "AppleGothic", "Noto Sans KR", "NanumGothic"]:
    try:
        font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.family"] = cand; break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False

# 팔레트 (발표 덱과 동일)
BLUE = "#3182F6"; RED = "#F04452"; INK = "#191F28"
GRAY = "#8B95A1"; LGRAY = "#E5E8EB"


def newfig(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    return fig, ax


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=160, bbox_inches="tight",
                facecolor="white")
    plt.close(fig); print("saved", os.path.join("docs", name))


# ============================================================
# 1. 모델 비교 — RF / CatBoost / AdaBoost (R² + Macro F1)
# ============================================================
mm = pd.read_csv(os.path.join(EXP, "model_metrics_summary.csv")).set_index("model_name")
order = ["random_forest", "catboost", "adaboost"]
disp = ["Random Forest", "CatBoost", "AdaBoost"]
r2 = [mm.loc[m, "r2_score"] for m in order]
f1 = [mm.loc[m, "macro_f1"] for m in order]

fig, ax = newfig(8.6, 4.6)
x = np.arange(len(order)); w = 0.36
r2cols = [BLUE, LGRAY, RED]            # AdaBoost 음수 강조
f1cols = [BLUE, LGRAY, LGRAY]
b1 = ax.bar(x - w/2, r2, w, color=r2cols, zorder=3, label="R²  (랭킹 정확도)")
b2 = ax.bar(x + w/2, f1, w, color=f1cols, zorder=3, label="Macro F1  (분류)")
for r, v in zip(b1, r2):
    ax.text(r.get_x()+r.get_width()/2, v + (0.018 if v >= 0 else -0.055),
            f"{v:.2f}", ha="center", fontsize=12, fontweight="bold",
            color=(RED if v < 0 else INK))
for r, v in zip(b2, f1):
    ax.text(r.get_x()+r.get_width()/2, v + 0.018, f"{v:.2f}",
            ha="center", fontsize=12, fontweight="bold", color=INK)
ax.axhline(0, color=INK, lw=1.0, zorder=2)
ax.text(0, max(f1)*1.16, "채택", ha="center", fontsize=11.5, fontweight="bold", color=BLUE)
ax.annotate("R² 음수 → 순위 추천에 부적합", xy=(2 - w/2, r2[2]),
            xytext=(2, -0.30), ha="center", fontsize=10.5, color=RED, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(disp, fontsize=12.5, color=INK)
ax.set_ylim(min(r2)*1.45, max(f1)*1.32); ax.set_yticks([])
ax.legend(frameon=False, fontsize=10.5, loc="upper right")
save(fig, "fig_model_comparison.png")


# ============================================================
# 2. 피처 중요도 top 10 — 별점(rating)은 6위에 불과
# ============================================================
with open(os.path.join(EXP, "feature_importance.json"), encoding="utf-8") as f:
    fi = json.load(f)
imp = fi["overall_all"]
KO = {
    "taste_confidence": "맛 신뢰도", "seat_group": "단체석",
    "collected_review_count": "수집 리뷰수", "spaciousness_confidence": "공간감 신뢰도",
    "review_count": "리뷰수", "rating": "별점", "spaciousness_score": "공간감",
    "value_score": "가성비", "noise_score": "소음", "taste_score": "맛",
}
top = list(imp.items())[:10]
names = [KO.get(k, k) for k, _ in top]
vals = [v for _, v in top]
cols = [RED if k == "rating" else BLUE for k, _ in top]  # 별점만 빨강

fig, ax = newfig(8.4, 5.0)
y = np.arange(len(top))[::-1]
ax.barh(y, vals, color=cols, zorder=3, height=0.66)
for yi, v, (k, _) in zip(y, vals, top):
    ax.text(v + 0.003, yi, f"{v:.3f}", va="center", fontsize=11,
            fontweight=("bold" if k == "rating" else "normal"),
            color=(RED if k == "rating" else INK))
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=12, color=INK)
ax.set_xticks([])
ax.set_xlim(0, max(vals)*1.16)
ax.text(max(vals)*0.99, 4.0, "별점은 6위 — 리뷰에서 뽑은\n신호가 상위를 차지",
        ha="right", va="center", fontsize=11, color=RED, fontweight="bold")
save(fig, "fig_feature_importance.png")


# ============================================================
# 3. 피처 ablation — 빼면 성능이 떨어지는 = 중요한 피처
# ============================================================
ab = pd.read_csv(os.path.join(EXP, "feature_ablation.csv"))
ab = ab[ab["experiment"] != "baseline"].copy()
ABKO = {
    "drop_collected_review_count": "수집 리뷰수",
    "drop_collection_metadata": "수집 메타데이터(별점·리뷰수)",
    "drop_brightness_mentions": "조도 언급량",
    "drop_brightness_confidence": "조도 신뢰도",
    "drop_review_count": "리뷰수",
    "drop_portion_mentions": "양 언급량",
    "drop_value_confidence": "가성비 신뢰도",
    "drop_non_significant_group": "비유의 피처 묶음",
    "drop_rating": "별점",
    "drop_taste_score": "맛 점수",
}
ab["label"] = ab["experiment"].map(lambda e: ABKO.get(e, e))
ab = ab.sort_values("delta_macro_f1")
labels = ab["label"].tolist()
deltas = ab["delta_macro_f1"].tolist()
cols = [RED if d < 0 else GRAY for d in deltas]  # 빼서 떨어지면(빨강) 중요

fig, ax = newfig(8.6, 5.0)
y = np.arange(len(labels))[::-1]
ax.barh(y, deltas, color=cols, zorder=3, height=0.64)
for yi, d in zip(y, deltas):
    ax.text(d + (0.0008 if d >= 0 else -0.0008), yi, f"{d:+.3f}",
            va="center", ha=("left" if d >= 0 else "right"),
            fontsize=10, color=(RED if d < 0 else GRAY))
ax.axvline(0, color=INK, lw=1.0, zorder=2)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=11, color=INK)
ax.set_xticks([])
pad = max(abs(min(deltas)), abs(max(deltas))) * 1.45
ax.set_xlim(-pad, pad)
ax.set_title("피처 제거 시 Macro F1 변화  (← 떨어질수록 중요한 피처)",
             fontsize=12, color=INK, pad=12)
save(fig, "fig_ablation.png")


# ============================================================
# 4. 출처 편향 — 카카오↔네이버 교차검증 시 R² 붕괴
# ============================================================
with open(os.path.join(EXP, "source_aware_evaluation.json"), encoding="utf-8") as f:
    sa = json.load(f)
smap = {d["evaluation"]: d for d in sa}
keys = ["mixed_source_oof_reference", "train_kakao_test_naver", "train_naver_test_kakao"]
disp = ["혼합 출처\n(실제 운영)", "카카오 학습\n→ 네이버 검증", "네이버 학습\n→ 카카오 검증"]
r2 = [smap[k]["r2_score"] for k in keys]
cols = [BLUE, RED, RED]

fig, ax = newfig(8.2, 4.8)
b = ax.bar(disp, r2, color=cols, width=0.58, zorder=3)
for r, v in zip(b, r2):
    ax.text(r.get_x()+r.get_width()/2, v + (0.12 if v >= 0 else -0.18),
            f"{v:.2f}", ha="center", fontsize=13, fontweight="bold",
            color=(BLUE if v >= 0 else RED))
ax.axhline(0, color=INK, lw=1.1, zorder=2)
ax.set_ylim(min(r2)*1.18, 1.0); ax.set_yticks([])
ax.tick_params(axis="x", labelsize=11.5, colors=INK)
ax.text(0.5, min(r2)*0.62,
        "출처가 달라지면 R²가 음수로 붕괴\n→ 모델이 식당보다 '출처별 글쓰기 습관'을 일부 학습",
        ha="center", fontsize=10.5, color=RED, fontweight="bold")
save(fig, "fig_source_bias.png")

print("\n완료 → docs/ 에 차트 4종 생성")
