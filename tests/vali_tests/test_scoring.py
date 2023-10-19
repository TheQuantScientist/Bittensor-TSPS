# developer: Taoshidev
# Copyright © 2023 Taoshi, LLC

import unittest

from tests.vali_tests.samples.testing_data import TestingData
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.exceptions.incorrect_prediction_size_error import IncorrectPredictionSizeError
from vali_objects.exceptions.min_responses_exception import MinResponsesException
from vali_objects.scoring.scoring import Scoring


class TestScoring(TestBase):

    def test_calculate_weighted_rmse(self):
        predictions = []
        actual = []

        for x in range(0, 100):
            predictions.append(x)
            actual.append(x - 0.05)

        weighted_rmse = Scoring.calculate_weighted_rmse(predictions, actual)
        self.assertEqual(weighted_rmse, 0.04999999999999829)

    def test_scale_scores(self):
        scaled_scores = Scoring.scale_scores(TestingData.SCORES)
        self.assertEqual(scaled_scores, TestingData.SCALED_SCORES)

    def test_simple_scale_scores(self):
        try:
            sample_scores = {'5F7GYJfDNRccc2ZTFXqjWVEQ96Vjv2yEsa1wzr3ULijD8Qhd': 34.54618500386289}
            scaled_scores = Scoring.simple_scale_scores(sample_scores)
        except Exception as e:
            self.assertIsInstance(e, MinResponsesException)
        sample_scores = {'5F7GYJfDNRccc2ZTFXqjWVEQ96Vjv2yEsa1wzr3ULijD8Qhd': 33.54618500386289,
                         '5F7GYJfDNRccc2ZTFXqjWVEQ96Vjv2yEsa1wzr3ULijD8Qhe': 34.54618500386289,
                         '5F7GYJfDNRccc2ZTFXqjWVEQ96Vjv2yEsa1wzr3ULijD8Qhf': 35.54618500386289,
                         '5F7GYJfDNRccc2ZTFXqjWVEQ96Vjv2yEsa1wzr3ULijD8Qhg': 36.54618500386289}
        scaled_scores_results = {'5F7GYJfDNRccc2ZTFXqjWVEQ96Vjv2yEsa1wzr3ULijD8Qhd': 1.0,
         '5F7GYJfDNRccc2ZTFXqjWVEQ96Vjv2yEsa1wzr3ULijD8Qhe': 0.6666666666666667,
         '5F7GYJfDNRccc2ZTFXqjWVEQ96Vjv2yEsa1wzr3ULijD8Qhf': 0.33333333333333337,
         '5F7GYJfDNRccc2ZTFXqjWVEQ96Vjv2yEsa1wzr3ULijD8Qhg': 0.0}

        scaled_scores = Scoring.simple_scale_scores(sample_scores)
        self.assertEqual(scaled_scores_results, scaled_scores)

    def test_calculate_directional_accuracy(self):
        predictions = [1, 2, 1, 3, 4, 5, 6, 7, 6, 7, 8]
        actual = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        da = Scoring.calculate_directional_accuracy(predictions, actual)
        self.assertEqual(da, 0.8)

    def test_score_response(self):
        def assert_preds_size_error(a, p):
            try:
                Scoring.score_response(a, p)
            except Exception as e:
                self.assertIsInstance(e, IncorrectPredictionSizeError)

        actual = []
        predictions = [1, 2]

        assert_preds_size_error(actual, predictions)

        actual = [1, 2]
        predictions = []

        assert_preds_size_error(actual, predictions)

        actual = []
        predictions = []

        assert_preds_size_error(actual, predictions)

        actual = [1]
        predictions = [1, 2]

        assert_preds_size_error(actual, predictions)

        actual = [1, 2]
        predictions = [1]

        assert_preds_size_error(actual, predictions)

        predictions = []
        actual = []

        for x in range(0, 100):
            predictions.append(x)
            actual.append(x - 0.05)

        score = Scoring.score_response(predictions, actual)
        self.assertEqual(score, 0.22360679774997513)


if __name__ == '__main__':
    unittest.main()