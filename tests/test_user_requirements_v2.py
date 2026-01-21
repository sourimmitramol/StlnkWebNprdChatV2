import logging
import os
import sys

import pandas as pd

# Set up path to import agents
sys.path.append(os.getcwd())

from agents.query_bank import clean_id, format_df, generic_attribute_lookup


def test_fixes_v2():
    print("Testing format_df robustness...")
    df_date = pd.DataFrame(
        {"container": ["C1", "C2"], "eta": [pd.to_datetime("2026-01-21"), pd.NaT]}
    )
    # Force object type to test fallback
    df_date["eta_obj"] = df_date["eta"].astype(object)

    print("Columns types:")
    print(df_date.dtypes)

    print("\nFormatted Date Table:")
    print(format_df(df_date))

    print("\nTesting quantity synonyms...")
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
            "shipped_qty_multiple": ["100 CTN"],
        }
    )

    synonyms = [
        "quantity",
        "cargo_count",
        "shipped quantity",
        "shipped_qty_multiple",
        "shipped_quantity",
    ]
    for syn in synonyms:
        print(f"\nLookup for synonym '{syn}':")
        res = generic_attribute_lookup(df_qty, "P1", syn)
        print(res)

    print("\nTesting detail quantity synonyms...")
    det_synonyms = [
        "detail quantity",
        "cargo_detail_count",
        "container detail quantity",
    ]
    for syn in det_synonyms:
        print(f"\nLookup for synonym '{syn}':")
        res = generic_attribute_lookup(df_qty, "P1", syn)
        print(res)


if __name__ == "__main__":
    test_fixes_v2()
