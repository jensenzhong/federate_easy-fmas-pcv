import unittest

import numpy as np


class FairCentralizedSplitTests(unittest.TestCase):
    def test_centralized_training_uses_federated_local_train_union(self):
        from src.utils import load_config
        from src.data_preprocessing import (
            load_centralized_datasets,
            load_federated_datasets_for_scene_c,
        )

        config = load_config("configs/config.yaml")
        centralized = load_centralized_datasets(config, config_key="scene_c")
        client_train_sets, client_val_sets, global_val_set, global_test_set, fed_preprocessor = (
            load_federated_datasets_for_scene_c(config)
        )

        self.assertEqual(len(centralized["train_df"]), sum(len(v) for v in client_train_sets.values()))
        self.assertEqual(len(centralized["local_val_df"]), sum(len(v) for v in client_val_sets.values()))
        self.assertEqual(len(centralized["val_df"]), len(global_val_set))
        self.assertEqual(len(centralized["test_df"]), len(global_test_set))
        self.assertEqual(centralized["training_scope"], "federated_local_train_union")
        self.assertTrue(np.allclose(
            centralized["preprocessor"].feature_scaler.mean_,
            fed_preprocessor.feature_scaler.mean_,
        ))
        self.assertTrue(np.allclose(
            centralized["preprocessor"].feature_scaler.scale_,
            fed_preprocessor.feature_scaler.scale_,
        ))


if __name__ == "__main__":
    unittest.main()
