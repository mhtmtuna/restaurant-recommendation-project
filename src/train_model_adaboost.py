from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    from .model_training_common import ROOT, train_and_save
except ImportError:  # Support `python src/train_model_adaboost.py`.
    from model_training_common import ROOT, train_and_save

MODEL_PATH = ROOT / "models" / "restaurant_recommender_adaboost.joblib"
REPORT_PATH = ROOT / "data" / "model_report_adaboost.json"
PREDICTIONS_PATH = ROOT / "data" / "restaurant_label_scores_adaboost.csv"


def estimator_factory():
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2, min_samples_leaf=3, random_state=42),
        n_estimators=200,
        learning_rate=0.5,
        random_state=42,
    )


def main():
    train_and_save(
        model_name="adaboost",
        estimator_factory=estimator_factory,
        model_path=MODEL_PATH,
        report_path=REPORT_PATH,
        predictions_path=PREDICTIONS_PATH,
    )


if __name__ == "__main__":
    main()
