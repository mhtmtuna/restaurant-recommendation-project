from sklearn.ensemble import RandomForestClassifier

from model_training_common import ROOT, train_and_save

MODEL_PATH = ROOT / "models" / "restaurant_recommender.joblib"
REPORT_PATH = ROOT / "data" / "model_report.json"
PREDICTIONS_PATH = ROOT / "data" / "restaurant_label_scores.csv"


def estimator_factory():
    return RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced_subsample",
    )


def main():
    train_and_save(
        model_name="random_forest",
        estimator_factory=estimator_factory,
        model_path=MODEL_PATH,
        report_path=REPORT_PATH,
        predictions_path=PREDICTIONS_PATH,
    )


if __name__ == "__main__":
    main()
