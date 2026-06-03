try:
    from .model_training_common import ROOT, train_and_save
except ImportError:  # Support `python src/train_model_catboost.py`.
    from model_training_common import ROOT, train_and_save

MODEL_PATH = ROOT / "models" / "restaurant_recommender_catboost.joblib"
REPORT_PATH = ROOT / "data" / "model_report_catboost.json"
PREDICTIONS_PATH = ROOT / "data" / "restaurant_label_scores_catboost.csv"


def estimator_factory():
    try:
        from catboost import CatBoostClassifier
    except ImportError as error:
        raise ImportError(
            "CatBoost is not installed. Run `py -m pip install -r requirements.txt` "
            "or `py -m pip install catboost` before training this model."
        ) from error

    return CatBoostClassifier(
        iterations=300,
        depth=5,
        learning_rate=0.05,
        loss_function="Logloss",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )


def main():
    train_and_save(
        model_name="catboost",
        estimator_factory=estimator_factory,
        model_path=MODEL_PATH,
        report_path=REPORT_PATH,
        predictions_path=PREDICTIONS_PATH,
    )


if __name__ == "__main__":
    main()
