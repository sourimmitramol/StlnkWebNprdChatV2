import logging
import os
import sys

import pandas as pd

# Set up path to import agents
sys.path.append(os.getcwd())

from agents.query_bank import clean_id, format_df, generic_attribute_lookup


def test_fixes():
    # 1. Test clean_id
    print("Testing clean_id...")
    ids = ["PO#5302982894", "PO:123", "Container#C123", "  PO # 456  "]
    for i in ids:
        print(f"'{i}' -> '{clean_id(i)}'")

    # 2. Test format_df date formatting
    print("\nTesting format_df date formatting...")
    df = pd.DataFrame({"container": ["C1"], "eta": [pd.to_datetime("2026-01-21")]})
    print(format_df(df))

    # 3. Test quantity logic
    print("\nTesting quantity logic...")
    df_qty = pd.DataFrame(
        {
            "container_number": ["C1"],
            "po_number_multiple": ["P1"],
            "cargo_count": [100],
            "cargo_um": ["CTN"],
            "cargo_weight": [500.5],
            "cargo_meassure": [10.2],
            "cargo_detail_count": [10],
            "detail_cargo_um": ["PCS"],
            "ocean_bl_no_multiple": ["B1"],
        }
    )

    # Generic lookup for quantity
    print("Lookup for 'quantity':")
    res_qty = generic_attribute_lookup(df_qty, "P1", "quantity")
    print(res_qty)

    # Generic lookup for detail quantity
    print("\nLookup for 'detail quantity':")
    res_det = generic_attribute_lookup(df_qty, "P1", "detail quantity")
    print(res_det)


if __name__ == "__main__":
    test_fixes()
