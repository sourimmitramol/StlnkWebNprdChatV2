import difflib
import logging
import re

import pandas as pd

# Set up logging to see the output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_schema_guard")

# Mock COLUMN_SYNONYMS for testing
COLUMN_SYNONYMS = {
    "carrier name": "final_carrier_name",
    "arrival date": "eta_dp",
    "po": "po_number_multiple",
    "consignee": "consignee_code_multiple",
}


def resolve_col(col_name, valid_columns):
    # 1. Direct match
    if col_name in valid_columns:
        return col_name
    # 2. Case-insensitive match
    for vc in valid_columns:
        if vc.lower() == col_name.lower():
            return vc
    # 3. Synonym match
    normalized = col_name.lower().replace("_", " ").strip()
    if normalized in COLUMN_SYNONYMS:
        target = COLUMN_SYNONYMS[normalized]
        if target in valid_columns:
            return target
    # 4. Fuzzy match
    matches = difflib.get_close_matches(col_name, valid_columns, n=1, cutoff=0.7)
    if matches:
        return matches[0]
    return col_name


def test_repair():
    df = pd.DataFrame(
        columns=[
            "container_number",
            "final_carrier_name",
            "eta_dp",
            "po_number_multiple",
        ]
    )
    valid_columns = df.columns.tolist()

    test_cases = [
        {
            "code": "result = df[df['carrier name'] == 'MSC']['container_number'].tolist()",
            "expected_contains": "final_carrier_name",
        },
        {
            "code": "result = df[df['arrival date'] > '2025-01-01']['container_number'].iloc[0]",
            "expected_contains": "eta_dp",
        },
        {
            "code": "result = df[df['Container Number'] == 'XYZ']['po']",
            "expected_contains": "container_number",
            "expected_contains_alt": "po_number_multiple",
        },
        {
            "code": "result = df[df['contaner_num'] == '123']",  # Typo
            "expected_contains": "container_number",
        },
    ]

    for case in test_cases:
        code = case["code"]
        potential_cols = re.findall(r"['\"]([a-zA-Z0-9_ ]+?)['\"]", code)
        repaired_code = code

        for p_col in set(potential_cols):
            if p_col not in valid_columns:
                resolved = resolve_col(p_col, valid_columns)
                if resolved != p_col:
                    logger.info(f"REPAIR: '{p_col}' -> '{resolved}'")
                    repaired_code = re.sub(
                        f"(['\"]){re.escape(p_col)}(['\"])",
                        f"\\1{resolved}\\2",
                        repaired_code,
                    )

        logger.info(f"Original: {code}")
        logger.info(f"Repaired: {repaired_code}")

        assert case["expected_contains"] in repaired_code
        if "expected_contains_alt" in case:
            assert case["expected_contains_alt"] in repaired_code
        logger.info("PASS")
        logger.info("-" * 20)


if __name__ == "__main__":
    test_repair()
