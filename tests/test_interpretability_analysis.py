import unittest

import numpy as np

from interpretability_analysis import (
    assign_time_periods,
    budgeted_keep_count,
    cosine_similarity_matrix,
    key_horizon_indices,
    residual_summary,
)


class InterpretabilityAnalysisTests(unittest.TestCase):
    def test_key_horizon_indices_for_24_step_output(self):
        self.assertEqual(key_horizon_indices(24), [5, 11, 17, 23])

    def test_residual_summary_masks_non_positive_truth(self):
        y_true = np.array([10.0, 0.0, 20.0, -1.0])
        y_pred = np.array([12.0, 100.0, 17.0, 5.0])

        summary = residual_summary(y_true, y_pred)

        self.assertEqual(summary["count"], 2)
        self.assertAlmostEqual(summary["mean"], -0.5)
        self.assertAlmostEqual(summary["std"], 2.5)

    def test_assign_time_periods_uses_four_paper_friendly_groups(self):
        tod = np.array([0.02, 7 / 24, 12 / 24, 18 / 24, 23 / 24])

        labels = assign_time_periods(tod)

        self.assertEqual(
            labels.tolist(),
            ["Night", "Morning peak", "Off-peak", "Evening peak", "Night"],
        )

    def test_cosine_similarity_matrix_is_symmetric_with_unit_diagonal(self):
        tokens = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        sim = cosine_similarity_matrix(tokens)

        self.assertEqual(sim.shape, (3, 3))
        np.testing.assert_allclose(np.diag(sim), np.ones(3))
        np.testing.assert_allclose(sim, sim.T)
        self.assertAlmostEqual(sim[0, 1], 0.0)

    def test_budgeted_keep_count_uses_negative_budget_as_unlimited(self):
        self.assertEqual(budgeted_keep_count(batch_size=64, remaining=-1), 64)
        self.assertEqual(budgeted_keep_count(batch_size=64, remaining=10), 10)
        self.assertEqual(budgeted_keep_count(batch_size=64, remaining=0), 0)


if __name__ == "__main__":
    unittest.main()
