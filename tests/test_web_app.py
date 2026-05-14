import unittest
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.web_app import LABEL_COLUMNS_LIST, predict_scores_from_model, safe_json


class DummyModel:
    def predict_proba(self, x_data):
        return [[0.1 * (row + 1) + 0.01 * col for col in range(len(LABEL_COLUMNS_LIST))] for row in range(len(x_data))]


class WebAppSecurityTest(unittest.TestCase):
    def test_safe_json_escapes_script_breakout(self):
        payload = [{"restaurant_name": "</script><script>alert(1)</script>"}]
        encoded = safe_json(payload)

        self.assertNotIn("</script>", encoded.lower())
        self.assertIn("\\u003c", encoded)

    def test_predict_scores_uses_array_position_not_dataframe_index(self):
        bundle = {
            "model": DummyModel(),
            "numeric_features": ["rating"],
            "categorical_features": ["area"],
            "seat_columns": [],
        }
        features = pd.DataFrame(
            [
                {"restaurant_id": "a", "rating": 4.0, "area": "x", "seat_type": ""},
                {"restaurant_id": "b", "rating": 3.5, "area": "y", "seat_type": ""},
            ],
            index=[10, 20],
        )

        scores = predict_scores_from_model(bundle, features)

        self.assertAlmostEqual(scores["a"]["couple_meal_score"], 0.1)
        self.assertAlmostEqual(scores["b"]["couple_meal_score"], 0.2)


if __name__ == "__main__":
    unittest.main()
