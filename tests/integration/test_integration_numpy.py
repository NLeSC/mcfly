import pytest
import unittest

from integration import IntegrationSuite


class NumpyIntegrationSuite(IntegrationSuite):

    @pytest.mark.integration
    def test_integration(self):
        self.run_integration()


if __name__ == "__main__":
    unittest.main()
