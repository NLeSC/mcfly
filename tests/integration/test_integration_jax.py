import keras
import pytest
import unittest

from integration import IntegrationSuite


class JaxIntegrationSuite(IntegrationSuite):

    @unittest.skipIf(keras.backend.backend() != "jax", reason="requires keras backend jax")
    @pytest.mark.jax
    @pytest.mark.integration
    def test_integration(self):
        assert keras.backend.backend() == "jax", "Unexpected keras backend."
        self.run_integration()


if __name__ == "__main__":
    unittest.main()
