import unittest

import numpy as np

from visualize_forecast import (
    compute_selected_horizon_node_metrics,
    select_representative_node,
)


class VisualizeResultAnalysisTests(unittest.TestCase):
    def test_compute_selected_horizon_node_metrics_uses_one_based_horizons(self):
        y_true = np.array([
            [[10.0, 20.0, 40.0], [5.0, 10.0, 20.0]],
            [[20.0, 30.0, 60.0], [10.0, 20.0, 40.0]],
        ])
        y_pred = y_true + np.array([
            [[1.0, -2.0, 4.0], [2.0, -1.0, 4.0]],
            [[-1.0, 3.0, -6.0], [-2.0, 2.0, -8.0]],
        ])

        metrics = compute_selected_horizon_node_metrics(y_true, y_pred, [1, 3])

        self.assertEqual(metrics["mae"].shape, (2, 2))
        np.testing.assert_allclose(metrics["mae"][:, 0], [1.0, 2.0])
        np.testing.assert_allclose(metrics["mae"][:, 1], [5.0, 6.0])
        np.testing.assert_allclose(metrics["rmse"][:, 0], [1.0, 2.0])
        np.testing.assert_allclose(metrics["mape"][:, 0], [0.075, 0.3])

    def test_selected_horizon_metrics_mask_non_positive_truth(self):
        y_true = np.array([
            [[0.0], [10.0]],
            [[20.0], [-1.0]],
        ])
        y_pred = np.array([
            [[100.0], [12.0]],
            [[24.0], [100.0]],
        ])

        metrics = compute_selected_horizon_node_metrics(y_true, y_pred, [1])

        np.testing.assert_allclose(metrics["mae"][:, 0], [4.0, 2.0])
        np.testing.assert_allclose(metrics["mape"][:, 0], [0.2, 0.2])

    def test_select_representative_node_returns_median_error_node(self):
        per_node_mae = np.array([10.0, 2.0, 20.0, 5.0, 12.0])

        self.assertEqual(select_representative_node(per_node_mae), 0)


if __name__ == "__main__":
    unittest.main()
