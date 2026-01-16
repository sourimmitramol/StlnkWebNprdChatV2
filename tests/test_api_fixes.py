import os
import sys
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.append(os.getcwd())

# Mock the agent and other dependencies before importing api.main
with patch("agents.azure_agent.initialize_azure_agent") as mock_init:
    mock_agent = MagicMock()
    mock_llm = MagicMock()
    mock_init.return_value = (mock_agent, mock_llm)

    # Also mock get_shipment_df
    with patch("services.azure_blob.get_shipment_df") as mock_get_df:
        import pandas as pd

        mock_get_df.return_value = pd.DataFrame(
            {
                "consignee_code_multiple": ["0000866"],
                "container_number": ["CONT1234567"],
            }
        )

        # Now we can import api.main
        import api.main

        api.main.AGENT = mock_agent  # Set the global AGENT
        from api.main import ask
        from api.schemas import QueryWithConsigneeBody

        def test_ask_fix():
            print("Running test_ask_fix...")
            body = QueryWithConsigneeBody(
                question="What is starlink",
                consignee_code="0000866",
                session_id="test_session",
            )

            # Mock agent.invoke to return a simple result
            mock_agent.invoke.return_value = {
                "output": "Starlink is a satellite constellation.",
                "intermediate_steps": [],
            }

            try:
                # Test the fix for UnboundLocalError
                response = ask(body)
                print("Test passed! Response obtained without UnboundLocalError.")
                print(f"Response: {response['response']}")
            except UnboundLocalError as e:
                print(f"Test failed! UnboundLocalError encountered: {e}")
            except NameError as e:
                print(f"NameError: {e}. Check if AGENT or LLM are defined in api.main")
            except Exception as e:
                print(f"An unexpected error occurred: {type(e).__name__}: {e}")
                import traceback

                traceback.print_exc()

        if __name__ == "__main__":
            test_ask_fix()
