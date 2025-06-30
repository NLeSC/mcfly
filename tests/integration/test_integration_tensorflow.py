import keras
import pytest
import unittest

from integration import IntegrationSuite


class TensorflowIntegrationSuite(IntegrationSuite):

    @unittest.skipIf(keras.backend.backend() != "tensorflow", reason="requires keras backend tensorflow")
    @pytest.mark.tensorflow
    @pytest.mark.integration
    def test_integration(self):
        assert keras.backend.backend() == "tensorflow", "Unexpected keras backend."
        self.run_integration()

    def generate_random_data_sets(self):
        import tensorflow as tf

        X_train, X_val, y_train, y_val, num_classes, x_shape = super().generate_random_data_sets()
        return (
            tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size=20),
            tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size=20),
            None, None,
            num_classes, x_shape
        )


if __name__ == '__main__':
    unittest.main()
