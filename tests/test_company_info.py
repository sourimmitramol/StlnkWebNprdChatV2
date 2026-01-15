import os
import sys

# Add the project root to sys.path
sys.path.append(os.getcwd())

import logging

from agents.tools import handle_non_shipping_queries

# Set up logging to avoid noise
logging.basicConfig(level=logging.ERROR)


def test_company_info():
    test_queries = [
        "Who is the CEO of MCS America?",
        "What is their mission statement?",  # Follow up
        "Where is your office in Egypt?",
        "Tell me about your core values",
        "Hi there, how are you today?",
    ]

    print("--- Testing Company Overview Knowledge ---\n")
    for q in test_queries:
        print(f"Query: {q}")
        response = handle_non_shipping_queries(q)
        print(f"Response: {response}\n")


if __name__ == "__main__":
    test_company_info()
