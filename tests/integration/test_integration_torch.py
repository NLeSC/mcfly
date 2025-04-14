import keras
import pytest
import unittest

from integration import IntegrationSuite


class PyTorchIntegrationSuite(IntegrationSuite):

    @unittest.skipIf(keras.backend.backend() != "torch", reason="requires keras backend torch")
    @pytest.mark.torch
    @pytest.mark.integration
    def test_integration(self):
        assert keras.backend.backend() == "torch", "Unexpected keras backend."
        self.run_integration()

    def generate_random_data_sets(self):
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        X_train, X_val, y_train, y_val, num_classes, x_shape = super().generate_random_data_sets()
        return (
            DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=20),
            DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=20),
            None, None,
            num_classes, x_shape
        )


if __name__ == '__main__':
    unittest.main()
