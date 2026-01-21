import logging
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from agents.analytics_engine import ShipmentAnalyst


class TestAnalystIntegration(unittest.TestCase):
    def setUp(self):
        # We need to mock settings and other things if we initialize the real class
        # but let's try to just patch the parts we need.
        with patch("agents.analytics_engine.AzureChatOpenAI"):
            self.analyst = ShipmentAnalyst()

        # Mock the LLM
        self.analyst.llm = MagicMock()

    @patch("agents.analytics_engine.match_query_bank")
    def test_schema_guard_integration(self, mock_bank):
        mock_bank.return_value = None  # Force code generation

        # Sample data
        df = pd.DataFrame(
            {
                "container_number": ["ABCD123"],
                "final_carrier_name": ["MSC"],
                "consignee_code_multiple": ["001"],
            }
        )

        # Mock response
        mock_response = MagicMock()
        mock_response.content = (
            "result = df[df['carrier name'] == 'MSC']['container_number'].iloc[0]"
        )
        self.analyst.llm.invoke.return_value = mock_response

        result = self.analyst.analyze("Which container is for MSC?", df)

        print(f"Result: {result}")
        self.assertEqual(result, "ABCD123")

    @patch("agents.analytics_engine.match_query_bank")
    def test_schema_guard_typo(self, mock_bank):
        mock_bank.return_value = None

        df = pd.DataFrame(
            {"container_number": ["XYZ999"], "consignee_code_multiple": ["001"]}
        )

        mock_response = MagicMock()
        mock_response.content = (
            "result = df[df['contaner_num'] == 'XYZ999']['container_number'].iloc[0]"
        )
        self.analyst.llm.invoke.return_value = mock_response

        result = self.analyst.analyze("Check XYZ999", df)

        print(f"Result: {result}")
        self.assertEqual(result, "XYZ999")


if __name__ == "__main__":
    unittest.main()
