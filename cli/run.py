# cli/run.py
import sys
import logging
import pandas as pd

from agents.azure_agent import initialize_azure_agent
from agents.tools import TOOLS
from utils.logger import logger

def run_azure_chatbot_cli() -> None:
    """Terminal‑only interface – exactly the `run_azure_chatbot_cli` you had."""
    print("\n=== Azure Shipping Chatbot (CLI) ===")
    print("Type 'exit' / 'quit' to stop.\n")

    try:
        agent, _ = initialize_azure_agent(TOOLS)
        logger.info("Agent initialized for CLI")
    except Exception as exc:
        logger.error(f"Failed to start agent: {exc}")
        sys.exit(1)

    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit", "bye"}:
            print("\nGood‑bye!")
            break

        if not user:
            continue

        # Very small “greeting” shortcut
        if user.lower().startswith(("hi", "hello", "hey")):
            print("Bot: Hello! How can I help you with the shipment data?")
            continue

        try:
            answer = agent.invoke(user)  # Use .invoke instead of .run
            print(f"Bot: {answer}")
        except Exception as exc:
            logger.exception("Agent failed")
            print(f"Bot: ⚠️  I ran into an error – try re‑phrasing. ({exc})")

        # Example of using pandas to process data
        df = pd.DataFrame({"your_column": ["2021-01-01", "invalid_date"]})
        df["your_column"] = pd.to_datetime(df["your_column"], errors="coerce")
        # Now you can use .dt safely:
        df["your_column"].dt.day


if __name__ == "__main__":
    run_azure_chatbot_cli()
