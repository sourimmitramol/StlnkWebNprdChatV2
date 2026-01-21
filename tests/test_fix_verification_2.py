import logging
from unittest.mock import patch

import pandas as pd

# Disable logging for cleaner output
logging.disable(logging.CRITICAL)

with patch("agents.tools.get_shipment_df") as mock_df:
    mock_df.return_value = pd.DataFrame(
        {
            "container_number": ["C1"],
            "po_number_multiple": ["P1"],
            "ocean_bl_no_multiple": ["B1"],
            "consignee_code_multiple": ["0005171"],
        }
    )

    from agents.prompts import map_synonym_to_column
    from agents.query_bank import generic_attribute_lookup
    from agents.tools import _df_filtered_by_consignee

    def test_consignee_filtering():
        df = pd.DataFrame(
            {
                "container_number": ["C1", "C2"],
                "consignee_code_multiple": ["0005171", "5172"],
            }
        )
        with patch("agents.tools.get_shipment_df", return_value=df):
            # Test 1: Padded code matches padded DB
            res = _df_filtered_by_consignee(["MCS(0005171)"])
            print(f"Test 1 (Padded match): {len(res)} rows (Expect 1)")

            # Test 2: Padded code matches unpadded DB
            res = _df_filtered_by_consignee(["MCS(0005172)"])
            print(
                f"Test 2 (Padded code matches unpadded DB): {len(res)} rows (Expect 1)"
            )

    def test_column_mapping():
        print(
            f"Mapping 'po_number_multiple': {map_synonym_to_column('po_number_multiple')} (Expect po_number_multiple)"
        )
        print(
            f"Mapping 'po number': {map_synonym_to_column('po number')} (Expect po_number_multiple)"
        )

    def test_attribute_lookup():
        df = pd.DataFrame(
            {
                "container_number": ["C1"],
                "po_number_multiple": ["5302999796, 5303014497"],
                "ocean_bl_no_multiple": ["BL1"],
            }
        )
        # Mock map_synonym_to_column to return what we expect
        res = generic_attribute_lookup(df, "5302999796", "container_number")
        print(
            f"Attribute Lookup Result (Valid Match): {'found' if 'C1' in str(res) else 'not found'}"
        )

    if __name__ == "__main__":
        test_consignee_filtering()
        test_column_mapping()
        test_attribute_lookup()
