import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATHS = [
    ROOT / "data" / "model_report.json",
    ROOT / "data" / "model_report_adaboost.json",
    ROOT / "data" / "model_report_catboost.json",
]
OUT_PATH = ROOT / "data" / "model_comparison.json"


def load_report(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    evaluation = report.get("evaluation", {})
    return {
        "model_name": report.get("model_name", path.stem),
        "report_path": str(path.relative_to(ROOT)),
        "r2_score": evaluation.get("r2_score"),
        "macro_f1": evaluation.get("macro_f1"),
        "micro_f1": evaluation.get("micro_f1"),
        "samples_f1": evaluation.get("samples_f1"),
    }


def main():
    rows = [row for row in (load_report(path) for path in REPORT_PATHS) if row]
    if not rows:
        raise FileNotFoundError("No model reports found. Train at least one model first.")

    rows = sorted(rows, key=lambda row: (row["r2_score"] is not None, row["r2_score"]), reverse=True)
    comparison = {
        "ranking_metric": "r2_score",
        "best_model": rows[0]["model_name"],
        "models": rows,
    }

    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print("model comparison")
    for idx, row in enumerate(rows, start=1):
        r2 = f"{row['r2_score']:.4f}" if row["r2_score"] is not None else "n/a"
        macro_f1 = f"{row['macro_f1']:.4f}" if row["macro_f1"] is not None else "n/a"
        micro_f1 = f"{row['micro_f1']:.4f}" if row["micro_f1"] is not None else "n/a"
        print(
            f"{idx}. {row['model_name']}: "
            f"r2={r2}, "
            f"macro_f1={macro_f1}, "
            f"micro_f1={micro_f1}"
        )
    print(f"best model: {comparison['best_model']}")
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
