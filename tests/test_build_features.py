import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.build_features import count_sentiment, label_value, matches_seat_type


class BuildFeaturesTest(unittest.TestCase):
    def test_negation_window_handles_spaced_korean_expression(self):
        positive, negative = count_sentiment(
            "분위기는 좋지만 다시 먹고 싶을 만큼 맛있지는 않아요",
            positive_words=["맛있"],
            negative_words=["맛없"],
            negation_prefixes=["안", "않"],
        )

        self.assertFalse(positive)
        self.assertTrue(negative)

    def test_negated_negative_counts_as_positive(self):
        positive, negative = count_sentiment(
            "생각보다 맛없지는 않아요",
            positive_words=["맛있"],
            negative_words=["맛없"],
            negation_prefixes=["안", "않", "없지는"],
        )

        self.assertTrue(positive)
        self.assertFalse(negative)

    def test_seat_regex_matches_two_person_table(self):
        self.assertTrue(matches_seat_type("2인용 테이블이 많아요", "couple", []))

    def test_label_value_ignores_negated_label_keyword(self):
        self.assertEqual(label_value(["데이트 아님"], ["데이트"], ["아님"]), 0)


if __name__ == "__main__":
    unittest.main()
