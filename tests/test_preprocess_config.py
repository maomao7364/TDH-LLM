import ast
import numpy as np
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _module_ast(name):
    return ast.parse((ROOT / name).read_text(encoding="utf-8"))


def _find_arg_default(tree, option_name):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and first.value == option_name):
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                return ast.literal_eval(keyword.value)
    raise AssertionError(f"{option_name} default not found")


class PreprocessConfigTest(unittest.TestCase):
    def test_custom_preprocess_defaults_to_24_input_steps(self):
        tree = _module_ast("preprocess_custom.py")
        self.assertEqual(_find_arg_default(tree, "--input-len"), 24)

    def test_custom_preprocess_splits_after_dropping_first_day(self):
        source = (ROOT / "preprocess_custom.py").read_text(encoding="utf-8")
        self.assertIn("WEEKLY_OFFSET = STEPS_PER_DAY * 7", source)
        self.assertIn("weekly = np.zeros_like(flow_3d)", source)
        self.assertIn("weekly[WEEKLY_OFFSET:] = flow_3d[:-WEEKLY_OFFSET]", source)
        self.assertIn("data = data[WEEKLY_OFFSET:]", source)
        self.assertIn("train_end = int(T_new * TRAIN_RATIO)", source)
        self.assertIn("val_end = int(T_new * (TRAIN_RATIO + VAL_RATIO))", source)
        self.assertNotIn("train_end = int(T * TRAIN_RATIO)", source)

    def test_pems_preprocess_uses_24_input_steps_and_post_drop_split(self):
        source = (ROOT / "preprocess_pems.py").read_text(encoding="utf-8")
        self.assertIn("INPUT_LEN = 24", source)
        self.assertIn("OUTPUT_LEN = 24", source)
        self.assertIn("WEEKLY_OFFSET = STEPS_PER_DAY * 7", source)
        self.assertIn("weekly = np.zeros_like(flow)", source)
        self.assertIn("weekly[WEEKLY_OFFSET:] = flow[:-WEEKLY_OFFSET]", source)
        self.assertIn("data = data[WEEKLY_OFFSET:]", source)
        self.assertIn("train_end = int(T_new * TRAIN_RATIO)", source)
        self.assertIn("val_end = int(T_new * (TRAIN_RATIO + VAL_RATIO))", source)


class DatasetShapeValidationTest(unittest.TestCase):
    def _data(self, x_shape=(2, 24, 50, 5), y_shape=(2, 24, 50, 3)):
        x = np.zeros(x_shape, dtype=np.float32)
        y = np.zeros(y_shape, dtype=np.float32)
        return {
            "x_train": x,
            "y_train": y,
            "x_val": x,
            "y_val": y,
            "x_test": x,
            "y_test": y,
        }

    def test_validate_dataset_shapes_accepts_matching_shapes(self):
        from dataset_checks import validate_dataset_shapes

        info = validate_dataset_shapes(
            self._data(), input_len=24, output_len=24, input_dim=5,
            num_nodes=50, target_dim=3,
        )
        self.assertEqual(info["input_len"], 24)
        self.assertEqual(info["input_dim"], 5)
        self.assertEqual(info["target_dim"], 3)

    def test_validate_dataset_shapes_rejects_input_len_mismatch(self):
        from dataset_checks import validate_dataset_shapes

        with self.assertRaisesRegex(ValueError, "input_len"):
            validate_dataset_shapes(
                self._data(x_shape=(2, 12, 50, 4)),
                input_len=24,
                output_len=24,
                input_dim=5,
                num_nodes=50,
                target_dim=3,
            )

    def test_validate_dataset_shapes_rejects_target_dim_mismatch(self):
        from dataset_checks import validate_dataset_shapes

        with self.assertRaisesRegex(ValueError, "target_dim"):
            validate_dataset_shapes(
                self._data(y_shape=(2, 24, 50, 2)),
                input_len=24,
                output_len=24,
                input_dim=5,
                num_nodes=50,
                target_dim=3,
            )


if __name__ == "__main__":
    unittest.main()
