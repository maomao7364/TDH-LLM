import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from interpretability_analysis import (
    assign_time_periods,
    budgeted_keep_count,
    cosine_similarity_matrix,
    key_horizon_indices,
    redraw_saved_figures,
    residual_statistics_label,
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

    def test_residual_statistics_label_uses_math_symbols_and_all_metrics(self):
        summary = {
            "mean": -0.224,
            "std": 31.234,
            "mae": 19.294,
            "rmse": 31.235,
        }

        label = residual_statistics_label(summary)

        self.assertEqual(
            label,
            r"$\mu$=-0.22   $\sigma$=31.23   MAE=19.29   RMSE=31.23",
        )

    def test_redraw_saved_figures_creates_refined_residual_and_gmsf_outputs(self):
        rng = np.random.default_rng(7)
        y_true = np.full((12, 3, 24), 100.0, dtype=np.float64)
        horizon_scale = np.linspace(1.0, 2.0, 24)[None, None, :]
        y_pred = y_true + rng.normal(size=y_true.shape) * horizon_scale
        gate_df = pd.DataFrame(
            {
                "period": ["Night", "Morning peak", "Off-peak", "Evening peak"],
                "count": [100, 100, 100, 100],
                "reference_dependency": [0.39, 0.34, 0.30, 0.31],
                "trend_weight": [0.24, 0.24, 0.18, 0.24],
                "periodic_weight": [0.34, 0.38, 0.41, 0.39],
                "residual_weight": [0.42, 0.38, 0.41, 0.37],
            }
        )

        with TemporaryDirectory() as source_tmp, TemporaryDirectory() as out_tmp:
            source_dir = Path(source_tmp)
            out_dir = Path(out_tmp)
            np.savez_compressed(
                source_dir / "interpretability_predictions.npz",
                y_true=y_true,
                y_pred=y_pred,
            )
            gate_df.to_csv(source_dir / "gate_weights_by_time_period.csv", index=False)
            (out_dir / "residual_density_across_horizons.png").write_bytes(b"stale")
            (out_dir / "residual_quantile_fan_across_horizons.png").write_bytes(b"stale")

            redraw_saved_figures(source_dir, out_dir)

            self.assertTrue((out_dir / "residual_violin_by_horizon.png").exists())
            self.assertTrue((out_dir / "gmsf_component_weights_by_period.png").exists())
            self.assertFalse((out_dir / "residual_density_across_horizons.png").exists())
            self.assertFalse((out_dir / "residual_quantile_fan_across_horizons.png").exists())
            self.assertFalse((out_dir / "residual_violin_by_horizon.svg").exists())
            self.assertFalse((out_dir / "residual_violin_by_horizon.pdf").exists())
            self.assertFalse((out_dir / "residual_density_across_horizons.svg").exists())
            self.assertFalse((out_dir / "residual_density_across_horizons.pdf").exists())
            self.assertFalse((out_dir / "residual_quantile_fan_across_horizons.svg").exists())
            self.assertFalse((out_dir / "residual_quantile_fan_across_horizons.pdf").exists())
            self.assertFalse((out_dir / "gmsf_component_weights_by_period.svg").exists())
            self.assertFalse((out_dir / "gmsf_component_weights_by_period.pdf").exists())
            summary = pd.read_csv(out_dir / "residual_distribution_summary.csv")
            self.assertEqual(summary["minutes"].tolist(), [30, 60, 90, 120])
            saved_gate = pd.read_csv(out_dir / "gate_weights_by_time_period.csv")
            self.assertEqual(saved_gate["period"].tolist(), gate_df["period"].tolist())


if __name__ == "__main__":
    unittest.main()
