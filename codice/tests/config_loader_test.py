import os
import tempfile
import unittest

from config_loader import get_nested, load_config


class TestConfigLoader(unittest.TestCase):
    def test_load_config_none_restituisce_dizionario_vuoto(self):
        self.assertEqual(load_config(None), {})

    def test_load_config_yaml_e_get_nested(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False, encoding="utf-8") as file:
            file.write(
                "run:\n"
                "  use_saved_model: false\n"
                "feature_selection:\n"
                "  n_iter: 7\n"
            )
            path = file.name

        try:
            config = load_config(path)
            self.assertFalse(get_nested(config, "run.use_saved_model"))
            self.assertEqual(get_nested(config, "feature_selection.n_iter"), 7)
            self.assertEqual(get_nested(config, "missing.key", "fallback"), "fallback")
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
