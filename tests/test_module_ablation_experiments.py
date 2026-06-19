import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


class ModuleAblationExperimentTest(unittest.TestCase):
    def test_legacy_experiment_files_are_removed(self):
        legacy_files = [
            "_training.py",
            "ablation_models.py",
            "llm_backbones.py",
            "train_gpt2_random.py",
            "train_llama.py",
            "train_no_gmsf.py",
            "train_no_llm.py",
            "train_no_mtd.py",
            "train_no_tcdg.py",
            "train_qwen2.py",
        ]
        for filename in legacy_files:
            self.assertFalse((EXPERIMENTS / filename).exists(), f"{filename} should be removed")

    def test_module_model_declares_supported_variants(self):
        source = (EXPERIMENTS / "module_ablation_model.py").read_text(encoding="utf-8")
        self.assertIn("MODULE_VARIANTS", source)
        for variant in (
            "full_modules",
            "no_mtd",
            "no_spatial",
            "no_gmsf",
            "no_llm",
            "no_horizon_decoder",
        ):
            self.assertIn(f"'{variant}'", source)

    def test_entry_scripts_call_expected_variants(self):
        expected = {
            "train_full_modules.py": "full_modules",
            "train_no_mtd_current.py": "no_mtd",
            "train_no_spatial_current.py": "no_spatial",
            "train_no_gmsf_current.py": "no_gmsf",
            "train_no_llm_current.py": "no_llm",
            "train_no_horizon_decoder.py": "no_horizon_decoder",
        }
        for filename, variant in expected.items():
            path = EXPERIMENTS / filename
            self.assertTrue(path.exists(), f"{filename} missing")
            tree = ast.parse(path.read_text(encoding="utf-8"))
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
            self.assertTrue(
                any(
                    isinstance(call.func, ast.Name)
                    and call.func.id == "run_module_experiment"
                    and any(
                        kw.arg == "module_variant"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value == variant
                        for kw in call.keywords
                    )
                    for call in calls
                ),
                f"{filename} does not call run_module_experiment(module_variant='{variant}')",
            )

    def test_module_training_uses_current_daily_weekly_data_shape(self):
        source = (EXPERIMENTS / "module_ablation_training.py").read_text(encoding="utf-8")
        self.assertIn("target_dim=3", source)
        self.assertIn("input_dim=args.input_dim", source)
        self.assertIn("future_weekly", source)


if __name__ == "__main__":
    unittest.main()
