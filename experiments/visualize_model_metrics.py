"""Create a pandas-based comparison chart from trained model reports."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "model_comparison.json"
OUT_CSV = ROOT / "experiments" / "model_metrics_summary.csv"
OUT_PNG = ROOT / "experiments" / "model_metrics_comparison.png"


def main():
    comparison = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    metrics = pd.DataFrame(comparison["models"]).set_index("model_name")
    metrics.to_csv(OUT_CSV, encoding="utf-8-sig")

    axes = metrics[["r2_score", "macro_f1", "micro_f1", "samples_f1"]].plot(
        kind="bar",
        figsize=(11, 6),
        ylim=(-0.2, 0.8),
        rot=0,
        title="Model Metrics on Merged Kakao + Naver Dataset",
    )
    axes.axhline(0, color="black", linewidth=0.8)
    axes.set_xlabel("Model")
    axes.set_ylabel("Score")
    axes.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=160)

    print(f"saved: {OUT_CSV}")
    print(f"saved: {OUT_PNG}")
    print(metrics[["r2_score", "macro_f1", "micro_f1", "samples_f1"]].to_string())


if __name__ == "__main__":
    main()
