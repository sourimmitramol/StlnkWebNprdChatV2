import os
import sys

import pandas as pd

# Mock the dataframe
df = pd.DataFrame(
    {
        "container_number": ["CONT123", "CONT456"],
        "po_number_multiple": ["PO789", "PO012"],
        "hot_container_flag": ["Y", "N"],
        "delay_days": [5, 0],
        "ata_dp": [pd.Timestamp("2025-08-15"), pd.NaT],
        "eta_dp": [pd.Timestamp("2025-08-20"), pd.Timestamp("2025-08-25")],
        "etd_lp": [pd.Timestamp.now(), pd.Timestamp.now()],
        "load_port": ["SHANGHAI", "NINGBO"],
        "discharge_port": ["LOS ANGELES", "ROTTERDAM"],
        "final_vessel_name": ["MAERSK A", "CMA CGM B"],
        "booking_number_multiple": ["BKG001", "BKG002"],
        "supplier_vendor_name": ["Vendor A", "Vendor B"],
        "consignee_name_multiple": ["Consignee A", "Consignee B"],
    }
)

# Add the project directory to sys.path
sys.path.append(os.getcwd())

import logging

handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(handler)

# Also ensure our specific logger is active
logging.getLogger("shipping_chatbot").setLevel(logging.INFO)
logging.getLogger("shipping_chatbot").addHandler(handler)

from agents.query_bank import match_query_bank

test_queries = [
    "List out any delay shipments in August",
    "Show me the container list with hot flag",
    "What is the consignee of PO789",
    "will container CONT123 be moved via rail",
    "mother vessel details of BKG001",
    "booking number of PO012",
    "what containers depature from Shanghai today",
    "show po are on water",
]

print("--- Testing Query Bank ---")
for q in test_queries:
    print(f"Query: {q}")
    result = match_query_bank(q, df)
    if result:
        print(f"Result (first 50 chars): {result[:50]}...")
    else:
        print("Result: NO MATCH (Fallback to LLM)")
    print("-" * 20)
