import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


class ReferenceAblationExperimentTest(unittest.TestCase):
    def test_training_module_declares_supported_reference_modes(self):
        source = (EXPERIMENTS / "reference_ablation_training.py").read_text(encoding="utf-8")
        self.assertIn("REFERENCE_MODES", source)
        for mode in ("full_refs", "no_daily", "no_weekly", "no_refs"):
            self.assertIn(f"'{mode}'", source)

    def test_entry_scripts_call_expected_modes(self):
        expected = {
            "train_full_refs.py": "full_refs",
            "train_no_daily.py": "no_daily",
            "train_no_weekly.py": "no_weekly",
            "train_no_refs.py": "no_refs",
        }
        for filename, mode in expected.items():
            path = EXPERIMENTS / filename
            self.assertTrue(path.exists(), f"{filename} missing")
            tree = ast.parse(path.read_text(encoding="utf-8"))
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
            self.assertTrue(
                any(
                    isinstance(call.func, ast.Name)
                    and call.func.id == "run_reference_experiment"
                    and any(
                        kw.arg == "reference_mode"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value == mode
                        for kw in call.keywords
                    )
                    for call in calls
                ),
                f"{filename} does not call run_reference_experiment(reference_mode='{mode}')",
            )

    def test_training_module_masks_daily_and_weekly_channels(self):
        source = (EXPERIMENTS / "reference_ablation_training.py").read_text(encoding="utf-8")
        self.assertIn("x[:, 3, :, :] = 0", source)
        self.assertIn("x[:, 4, :, :] = 0", source)
        self.assertIn("future_daily = None", source)
        self.assertIn("future_weekly = None", source)


if __name__ == "__main__":
    unittest.main()
