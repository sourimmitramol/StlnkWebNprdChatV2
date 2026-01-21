import json
import logging
import re
from typing import Any, Callable, Dict, Optional

import pandas as pd

logger = logging.getLogger("shipping_chatbot")


def format_df(res_df: pd.DataFrame) -> str:
    """Helper to format dataframe without tabulate dependency. Formats dates as dd-mmm-yyyy."""
    try:
        if res_df is None or res_df.empty:
            return "No data found."
        temp_df = res_df.copy()
        # Format any datetime columns to dd-mmm-yyyy
        for col in temp_df.columns:
            if pd.api.types.is_datetime64_any_dtype(temp_df[col]):
                temp_df[col] = temp_df[col].dt.strftime("%d-%b-%Y")

        try:
            return temp_df.to_markdown(index=False)
        except:
            return temp_df.to_string(index=False)
    except Exception as e:
        logger.error(f"Error in format_df: {e}")
        return res_df.to_string(index=False)


def clean_id(val: str) -> str:
    """Removes symbols like #, :, etc. from identifiers."""
    if not val:
        return val
    val = val.strip()
    # Remove common prefixes like PO#, PO:, etc.
    cleaned = re.sub(r"^(?i:po|container|obl|bl)[#:\s]*", "", val)
    # Remove all remaining #, :, and internal spaces
    cleaned = re.sub(r"[#:\s]+", "", cleaned)
    return cleaned.strip()


def get_delayed_shipments(
    df: pd.DataFrame, month: Optional[str] = None, year: int = 2025, days: int = 0
) -> str:
    """Logic for Intent 7, 8, 29, 47"""
    temp_df = df.copy()

    # Ensure delay_days is calculated if not present
    if (
        "delay_days" not in temp_df.columns
        and "eta_dp" in temp_df.columns
        and "ata_dp" in temp_df.columns
    ):
        temp_df["delay_days"] = (temp_df["ata_dp"] - temp_df["eta_dp"]).dt.days

    mask = (temp_df["delay_days"] > days) & (temp_df["ata_dp"].notna())

    if month:
        # Convert month string to numeric
        try:
            month_num = pd.to_datetime(month, format="%B").month
        except:
            # Try short month
            try:
                month_num = pd.to_datetime(month, format="%b").month
            except:
                month_num = None

        if month_num:
            mask &= temp_df["ata_dp"].dt.month == month_num
            if year:
                mask &= temp_df["ata_dp"].dt.year == year

    result_df = temp_df[mask]
    if result_df.empty:
        return f"No delayed shipments found for {month or 'all time'} {year if month else ''}."

    cols = [
        "container_number",
        "po_number_multiple",
        "discharge_port",
        "eta_dp",
        "ata_dp",
        "delay_days",
    ]
    return format_df(result_df[cols])


def get_hot_shipments(
    df: pd.DataFrame, vessel: Optional[str] = None, month: Optional[str] = None
) -> str:
    """Logic for Intent 32, 48, 49, 50"""
    mask = (
        df["hot_container_flag"]
        .astype(str)
        .str.upper()
        .isin(["TRUE", "Y", "YES", "1", "HOT"])
    )

    if vessel:
        mask &= df["final_vessel_name"].str.contains(vessel, case=False, na=False) | df[
            "vessel_name"
        ].str.contains(vessel, case=False, na=False)

    result_df = df[mask]
    if result_df.empty:
        return "No HOT shipments found matching your criteria."

    return format_df(
        result_df[
            [
                "container_number",
                "po_number_multiple",
                "hot_container_flag",
                "final_vessel_name",
            ]
        ]
    )


def get_on_water_shipments(df: pd.DataFrame) -> str:
    """Logic for Intent 44: ata_dp is null, predicted/revised eta > today, status etc."""
    today = pd.Timestamp.now().normalize()
    # Logic: ata_dp == null and etd_lp <= today
    mask = (df["ata_dp"].isna()) & (df["etd_lp"].notna()) & (df["etd_lp"] <= today)

    result_df = df[mask]
    if result_df.empty:
        return "No shipments currently on water identified."

    return format_df(
        result_df[
            [
                "container_number",
                "po_number_multiple",
                "etd_lp",
                "eta_dp",
                "discharge_port",
            ]
        ]
    )


def get_rail_shipments(df: pd.DataFrame, container: Optional[str] = None) -> str:
    """Logic for Intent 26: Rail columns"""
    rail_cols = [
        "rail_load_dp_date",
        "rail_load_dp_lcn",
        "rail_departure_dp_date",
        "rail_departure_dp_lcn",
        "rail_arrival_destination_date",
        "rail_arrival_destination_lcn",
    ]

    if container:
        result_df = df[
            df["container_number"].str.contains(container, case=False, na=False)
        ]
    else:
        # Just show all where rail info exists
        result_df = df[df["rail_load_dp_date"].notna()]

    if result_df.empty:
        return "No rail movement information found."

    available_cols = [c for c in rail_cols if c in result_df.columns]
    column_list = ["container_number"] + available_cols
    return format_df(result_df[column_list])


def get_vessel_details(df: pd.DataFrame, po_or_id: str, type: str = "both") -> str:
    """Logic for Intent 37, 38"""
    mask = (
        df["po_number_multiple"].str.contains(po_or_id, case=False, na=False)
        | df["container_number"].str.contains(po_or_id, case=False, na=False)
        | df["booking_number_multiple"].str.contains(po_or_id, case=False, na=False)
    )

    result_df = df[mask]
    if result_df.empty:
        return f"No vessel details found for {po_or_id}."

    cols = ["container_number", "po_number_multiple", "final_vessel_name"]
    if type == "both":
        cols.append("first_vessel_name")

    return format_df(result_df[cols])


def get_booking_info(df: pd.DataFrame, search_val: str) -> str:
    """Logic for Intent 10, 11, 12, 33, 34, 35, 45, 46"""
    mask = (
        (df["container_number"].str.contains(search_val, case=False, na=False))
        | (df["po_number_multiple"].str.contains(search_val, case=False, na=False))
        | (df["booking_number_multiple"].str.contains(search_val, case=False, na=False))
    )

    result_df = df[mask]
    if result_df.empty:
        return f"No booking information found for {search_val}."

    cols = [
        "container_number",
        "po_number_multiple",
        "booking_number_multiple",
        "eta_dp",
    ]
    return format_df(result_df[cols])


def get_movement_stats(
    df: pd.DataFrame,
    direction: str = "departure",
    location: Optional[str] = None,
    country: Optional[str] = None,
    date_str: str = "today",
) -> str:
    """Logic for Intent 13, 14, 15, 16, 52"""
    try:
        date = (
            pd.Timestamp.now().normalize()
            if date_str == "today"
            else pd.to_datetime(date_str)
        )
    except:
        date = pd.Timestamp.now().normalize()

    col = "etd_lp" if direction == "departure" else "ata_dp"
    loc_col = "load_port" if direction == "departure" else "discharge_port"

    if col not in df.columns:
        return f"Date column {col} not found for statistical analysis."

    mask = df[col].dt.date == date.date()

    if location:
        mask &= df[loc_col].str.contains(location, case=False, na=False)
    if country:
        mask &= df[loc_col].str.contains(country, case=False, na=False)

    result_df = df[mask]
    count = len(result_df)

    if count == 0:
        return f"No {direction}s found for {location or country or 'all locations'} on {date.date()}."

    return (
        f"Total {direction}s for {location or country or 'all locations'} on {date.date()}: {count}\n\n"
        + format_df(result_df[["container_number", "po_number_multiple", col, loc_col]])
    )


def generic_attribute_lookup(df: pd.DataFrame, search_val: str, attr_col: str) -> str:
    """Logic for Intent 9, 17, 19, 20, 21, 23, 31, 36, 40, 41, 42. Handles specific quantity logic."""
    if not search_val:
        return "Please specify an identifier (PO, Container, or OBL) to lookup."

    # Clean the search identifier (remove #, symbols, prefixes)
    search_val = clean_id(search_val)

    # Map attribute synonym to actual column name
    from .prompts import map_synonym_to_column

    target_col = map_synonym_to_column(attr_col)

    # Check for special quantity logic
    is_detail_qty = attr_col.lower() in [
        "detail quantity",
        "cargo_detail_count",
        "container detail quantity",
    ]
    is_general_qty = attr_col.lower() in [
        "quantity",
        "cargo_count",
        "shipped quantity",
        "shipped_qty_multiple",
        "shipped_quantity",
    ]

    mask = (
        (
            df["container_number"]
            .astype(str)
            .str.contains(search_val, case=False, na=False)
        )
        | (
            df["po_number_multiple"]
            .astype(str)
            .str.contains(search_val, case=False, na=False)
        )
        | (
            df["po_number_multiple"]
            .astype(str)
            .apply(lambda x: search_val in [t.strip() for t in x.split(",")])
        )
        | (df["ocean_bl_no_multiple"].astype(str).str.contains(search_val, na=False))
    )

    result_df = df[mask].copy()
    if result_df.empty:
        logger.info(f"QUERY_BANK: attribute_lookup found NO records for '{search_val}'")
        return f"No records found matching identifier '{search_val}'."

    display_cols = ["container_number", "po_number_multiple"]

    if is_detail_qty:
        # concat(cargo_detail_count and detail_cargo_um)
        result_df["detail_quantity"] = (
            result_df["cargo_detail_count"].astype(str)
            + " "
            + result_df["detail_cargo_um"].astype(str)
        )
        display_cols.append("detail_quantity")
    elif is_general_qty:
        # concat(cargo_count and cargo_um) with extra col cargo_weight and cargo_meassure
        result_df["quantity"] = (
            result_df["cargo_count"].astype(str)
            + " "
            + result_df["cargo_um"].astype(str)
        )
        display_cols.extend(["quantity", "cargo_weight", "cargo_meassure"])
    else:
        if target_col not in df.columns:
            logger.warning(
                f"QUERY_BANK: attribute_lookup column '{target_col}' (from '{attr_col}') not found"
            )
            return f"Attribute field '{attr_col}' not found in the dataset."
        if target_col not in display_cols:
            display_cols.append(target_col)

    logger.info(
        f"QUERY_BANK: attribute_lookup found {len(result_df)} records for '{search_val}' (col: {target_col})"
    )
    return format_df(result_df[display_cols])


# Intent Registry
QUERY_BANK: Dict[str, Dict[str, Any]] = {
    "delayed_shipments": {
        "description": "Lists delayed or late shipments, can be filtered by month or min delay days.",
        "patterns": [
            r"list out any delay shipments in (\w+)",
            r"any containers delayed on (\w+)",
            r"delayed shipments on (\w+)",
            r"containers are delayed more than (\d+) days",
        ],
        "handler": get_delayed_shipments,
        "extract": lambda m: {
            "month": m.group(1) if not m.group(1).isdigit() else None,
            "days": int(m.group(1)) if m.group(1).isdigit() else 0,
        },
    },
    "hot_shipments": {
        "description": "Lists 'HOT' or priority containers, can be filtered by vessel or month.",
        "patterns": [
            r"show me the container list with hot flag",
            r"list the hot po in booking",
            r"what hot containers are shipping with ([\w\s]+) in (\w+)",
            r"report for ([\w\s]+) hot container",
        ],
        "handler": get_hot_shipments,
        "extract": lambda m: {"vessel": m.group(1) if len(m.groups()) > 0 else None},
    },
    "on_water": {
        "description": "Lists shipments currently on the water/in transit sea leg.",
        "patterns": [
            r"show po are on water",
            r"what is on water",
            r"containers on water",
        ],
        "handler": lambda df, **kwargs: get_on_water_shipments(df),
    },
    "rail_check": {
        "description": "Checks if a specific container is booked for rail movement.",
        "patterns": [r"will container ([\w\d]+) be moved via rail"],
        "handler": get_rail_shipments,
        "extract": lambda m: {"container": m.group(1)},
    },
    "vessel_details": {
        "description": "Provides vessel names (mother/first) for a PO or container.",
        "patterns": [
            r"mother vessel details of ([\w\d\-]+)",
            r"first and mother vessel details of ([\w\d\-]+)",
        ],
        "handler": get_vessel_details,
        "extract": lambda m: {
            "po_or_id": m.group(1),
            "type": "both" if "first" in m.string.lower() else "mother",
        },
    },
    "booking_lookup": {
        "description": "Finds booking number or status for a container or PO.",
        "patterns": [
            r"booking number of ([\w\d]+)",
            r"bkg ([\w\d]+)",
            r"what is the booking number of ([\w\d]+)",
            r"what is the status of mcs booking# ([\w\d]+)",
        ],
        "handler": get_booking_info,
        "extract": lambda m: {"search_val": m.group(1)},
    },
    "movement_stats": {
        "description": "Stats for arrivals/departures today at ports.",
        "patterns": [
            r"what containers departure from ([\w\s]+) today",
            r"how many will be departure today from ([\w\s]+)",
            r"how many containers will be departure today from ([\w\s]+)",
        ],
        "handler": get_movement_stats,
        "extract": lambda m: {"location": m.group(1), "direction": "departure"},
    },
    "attribute_lookup": {
        "description": "Generic lookup for specific fields (consignee, vendor, carrier, ready date, etc.)",
        "patterns": [
            (r"what is the consignee of\s+([\w\d\-#]+)", "consignee_name_multiple"),
            (r"cargo ready date for\s+([\w\d\-#]+)", "cargo_ready_date"),
            (r"shipped quantity for\s+([\w\d\-#]+)", "shipped_qty_multiple"),
            (
                r"what is the (?:shipper|supplier) for (?:po/container/obl#)?\s*([\w\d\-#]+)",
                "supplier_vendor_name",
            ),
            (r"which carrier (?:is handled it|handled it)", "final_carrier_name"),
            (r"please check the vendor name of\s+([\w\d\-#]+)", "supplier_vendor_name"),
            (r"what is discharge port eta of\s*([\w\d\-#]+)", "eta_dp"),
            (
                r"what is the container number for po#\s*([\w\d\-#]+)",
                "container_number",
            ),
            (
                r"container (?:carrying|for) (?:the |)po\s*([\w\d\-#]+)",
                "container_number",
            ),
            (r"what is po number in\s*([\w\d\-#]+)", "po_number_multiple"),
            (r"what is the quantity for\s*([\w\d\-#]+)", "quantity"),
            (r"quantity of\s*([\w\d\-#]+)", "quantity"),
            (r"what is the detail quantity for\s*([\w\d\-#]+)", "detail quantity"),
            (r"detail quantity of\s*([\w\d\-#]+)", "detail quantity"),
        ],
        "handler": generic_attribute_lookup,
        "extract": lambda m, attr: {
            "search_val": m.group(1).split()[-1] if m.groups() else None,
            "attr_col": attr,
        },
    },
    "status_tracking": {
        "description": "Checks status or milestones for a specific container or PO.",
        "patterns": [
            r"status of (?:the |)(?:po |container |)([\w\d\-]+)",
            r"where is (?:the |)(?:po |container |)([\w\d\-]+)",
            r"milestones of (?:the |)(?:po |container |)([\w\d\-]+)",
        ],
        "handler": lambda df, **kwargs: "NEEDS_MILESTONES:"
        + kwargs.get("search_val", ""),
        "extract": lambda m: {"search_val": m.group(1)},
    },
}

INTENT_CLASSIFIER_PROMPT = """
Analyze the user's shipping query and map it to one of these predefined intents.
If the query is technical (e.g., 'Filter rows where...', 'select attribute...'), treat it like a natural question.

INTENTS:
1. delayed_shipments: Queries about late or delayed containers/POs. Params: {{month, days, year}}
2. hot_shipments: Queries about 'HOT', priority, or urgent containers. Params: {{vessel, month}}
3. on_water: Shipments currently in transit on water. Params: {{}}
4. rail_check: Check container moved via rail. Params: {{container}}
5. vessel_details: Vessel names for PO or container. Params: {{po_or_id, type=['mother', 'both']}}
6. booking_lookup: Find booking numbers or status. Params: {{search_val}}
7. movement_stats: Arrival/departure counts at ports. Params: {{direction=['departure', 'arrival'], location, country, date_str}}
8. attribute_lookup: Specific field lookups. Params: {{search_val, attr_col}}
9. status_tracking: Status, location, or milestones for a specific PO/container ID. Params: {{search_val}}
   Mappings for attribute_lookup 'attr_col':
   - 'consignee' -> 'consignee_name_multiple'
   - 'ready date' / 'received' -> 'cargo_received_date_multiple'
   - 'shipped quantity' -> 'shipped_qty_multiple'
   - 'vendor' / 'supplier' / 'shipper' -> 'supplier_vendor_name'
   - 'carrier' -> 'final_carrier_name'
   - 'eta' / 'arrival date' -> 'eta_dp'
   - 'container number' -> 'container_number'
   - 'po number' -> 'po_number_multiple'

Return ONLY a JSON blob with "intent" and "params". 
Example: {{"intent": "attribute_lookup", "params": {{"search_val": "5302997239", "attr_col": "container_number"}}}}
If no intent matches exactly, return {{"intent": "none", "params": {{}}}}

User Query: {query}
Return ONLY valid JSON.
"""


def match_query_bank(
    query: str, df: pd.DataFrame, llm: Optional[Any] = None
) -> Optional[str]:
    """Matches query using Regex first, then LLM for robustness."""
    query_clean = query.lower().strip()

    # 1. Regex Match (Fast Path)
    for intent_name, entry in QUERY_BANK.items():
        for item in entry["patterns"]:
            pattern = item[0] if isinstance(item, tuple) else item
            attr = item[1] if isinstance(item, tuple) else None

            match = re.search(pattern, query_clean, re.IGNORECASE)
            if match:
                logger.debug(f"QUERY_BANK: Regex Match '{intent_name}'")
                try:
                    params = (
                        entry.get("extract", lambda m, a=None: {})(match, attr)
                        if attr
                        else entry.get("extract", lambda m: {})(match)
                    )
                    return entry["handler"](df, **params)
                except Exception as e:
                    logger.error(f"QUERY_BANK_ERR: {e}")

    # 2. LLM Match (Robust Path)
    if llm:
        try:
            logger.info("QUERY_BANK: Attempting LLM Intent Classification")
            resp = llm.invoke(INTENT_CLASSIFIER_PROMPT.format(query=query))
            content = resp.content.strip()

            # Robust JSON extraction
            if "```json" in content:
                content = content.split("```json")[-1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            data = json.loads(content)

            intent = data.get("intent")
            params = data.get("params", {})

            if intent in QUERY_BANK:
                logger.info(f"QUERY_BANK: LLM Matched '{intent}'")
                return QUERY_BANK[intent]["handler"](df, **params)
        except Exception as e:
            logger.error(f"QUERY_BANK_LLM_ERR: {e}")

    return None
