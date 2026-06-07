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
        "total_size": report.get("total_size"),
        "labels": report.get("labels", []),
        "r2_score": evaluation.get("r2_score"),
        "macro_f1": evaluation.get("macro_f1"),
        "micro_f1": evaluation.get("micro_f1"),
        "samples_f1": evaluation.get("samples_f1"),
    }


def is_compatible(row, reference):
    return (
        row.get("total_size") == reference.get("total_size")
        and row.get("labels") == reference.get("labels")
    )


def main():
    rows = [row for row in (load_report(path) for path in REPORT_PATHS) if row]
    if not rows:
        raise FileNotFoundError("No model reports found. Train at least one model first.")

    reference = rows[0]
    compatible_rows = []
    skipped_rows = []
    for row in rows:
        if is_compatible(row, reference):
            compatible_rows.append(row)
        else:
            skipped_rows.append(
                {
                    "model_name": row["model_name"],
                    "report_path": row["report_path"],
                    "reason": (
                        f"incompatible schema: total_size={row.get('total_size')}, "
                        f"labels={row.get('labels')}"
                    ),
                }
            )

    if not compatible_rows:
        raise ValueError("No compatible model reports found.")

    compatible_rows = sorted(
        compatible_rows,
        key=lambda row: (row["r2_score"] is not None, row["r2_score"]),
        reverse=True,
    )
    comparison = {
        "ranking_metric": "r2_score",
        "dataset_total_size": reference.get("total_size"),
        "labels": reference.get("labels", []),
        "best_model": compatible_rows[0]["model_name"],
        "models": compatible_rows,
        "skipped_reports": skipped_rows,
    }

    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print("model comparison")
    for idx, row in enumerate(compatible_rows, start=1):
        r2 = f"{row['r2_score']:.4f}" if row["r2_score"] is not None else "n/a"
        macro_f1 = f"{row['macro_f1']:.4f}" if row["macro_f1"] is not None else "n/a"
        micro_f1 = f"{row['micro_f1']:.4f}" if row["micro_f1"] is not None else "n/a"
        print(
            f"{idx}. {row['model_name']}: "
            f"r2={r2}, "
            f"macro_f1={macro_f1}, "
            f"micro_f1={micro_f1}"
        )
    for row in skipped_rows:
        print(f"skipped {row['model_name']}: {row['reason']}")
    print(f"best model: {comparison['best_model']}")
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
