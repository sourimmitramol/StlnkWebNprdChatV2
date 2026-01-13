# agents/tools.py
import logging
import re
from datetime import datetime, timedelta
from langchain_community.agent_toolkits import create_sql_agent
from config import settings
import pandas as pd
from fuzzywuzzy import process
from langchain.agents import Tool
from services.vectorstore import get_vectorstore
from utils.container import extract_container_number,extract_po_number,extract_ocean_bl_number
from utils.logger import logger
from services.azure_blob import get_shipment_df
from utils.misc import to_datetime, clean_container_number
from langchain_openai import AzureChatOpenAI
import sqlite3
from sqlalchemy import create_engine
import threading
from difflib import get_close_matches

from agents.prompts import (
    map_synonym_to_column,
    COLUMN_SYNONYMS,
    parse_time_period,           # Add this
    format_date_for_display,     # Add this
    is_date_in_range,
    map_intent_phrase            # Add this (if you use it)
)


def safe_sort_dataframe(df, sort_column, ascending=True):
    """Safe sorting compatible with all pandas versions"""
    if sort_column not in df.columns:
        return df

    try:
        # Try modern pandas first
        return df.sort_values(sort_column, ascending=ascending, na_position='last')
    except TypeError:
        # Fallback for older pandas
        df_copy = df.copy()
        df_copy['_temp_sort'] = pd.to_datetime(df_copy[sort_column], errors='coerce')
        df_sorted = df_copy.sort_values('_temp_sort', ascending=ascending)
        return df_sorted.drop('_temp_sort', axis=1)

# Helper – give every tool a clean copy of the DataFrame
def _df() -> pd.DataFrame:
    """
    Get filtered DataFrame based on thread-local consignee codes.
    Returns the full dataset if no consignee codes are set.
    """
    df = get_shipment_df()

    # Check if consignee codes are set in thread-local storage
    if hasattr(threading.current_thread(), 'consignee_code'):
        consignee_code = threading.current_thread().consignee_code

        if consignee_code and 'consignee_code_multiple' in df.columns:
            # Extract numeric codes from consignee codes
            numeric_codes = []
            for code in consignee_code:
                # Extract numeric part (e.g., "0045831" from "EDDIE BAUER LLC(0045831)")
                match = re.search(r'\((\d+)\)', code)
                if match:
                    numeric_codes.append(match.group(1))
                else:
                    # If already just numeric, use as is
                    numeric_codes.append(code.strip())

            # Filter by consignee codes
            pattern = r"|".join([rf"\b{re.escape(code)}\b" for code in numeric_codes])
            mask = df['consignee_code_multiple'].astype(str).apply(
                lambda x: bool(re.search(pattern, x))
            )
            filtered_df = df[mask]

            logger.debug(f"Filtered DataFrame: {len(df)} -> {len(filtered_df)} rows for consignee codes: {numeric_codes}")
            return filtered_df

    logger.debug(f"Returning unfiltered DataFrame with {len(df)} rows")
    return df


def handle_non_shipping_queries(query: str) -> str:
    """
    Handle greetings, thanks, small talk, and general non-shipping queries.
    Uses Azure OpenAI (AzureChatOpenAI) for general/knowledge-based questions.
    """

    q = query.lower().strip()

    # Quick responses for small-talk
    greetings = ["hi", "hello", "hey", "gm", "good morning", "good afternoon", "good evening", "hola"]
    if any(word in q for word in greetings):
        return "Hello! I’m MCS AI, your shipping assistant. How can I help you today?"

    thanks = ["thank", "thx", "thanks", "thank you", "ty", "much appreciated"]
    if any(word in q for word in thanks):
        return "You’re very welcome! Always happy to help. – MCS AI"

    if "how are you" in q or "how r u" in q:
        return "I’m doing great, thanks for asking! How about you? – MCS AI"

    if "who are you" in q or "your name" in q or "what is your name" in q:
        return "I’m MCS AI, your AI-powered shipping assistant. I can help you track containers, POs, and more."

    farewells = ["bye", "goodbye", "see you", "take care", "cya", "see ya"]
    if any(word in q for word in farewells):
        return "Goodbye! Have a wonderful day ahead. – MCS AI"

    # Detect non-shipping query
    shipping_keywords = ["container", "shipment", "cargo", "po", "eta", "vessel", "port", "delivery", "bill of lading"]
    if not any(word in q for word in shipping_keywords):

        try:
            # Initialize Azure Chat Model
            llm = AzureChatOpenAI(
                azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,   # your Azure model deployment name
                api_version=settings.AZURE_OPENAI_API_VERSION, # depends on your Azure setup
                temperature=0.8,
                max_tokens=800,
            )

            # Use LangChain message schema for clarity
            from langchain.schema import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content="You are MCS AI, a helpful and friendly assistant who answers general non-shipping questions concisely."),
                HumanMessage(content=query)
            ]

            response = llm.invoke(messages)
            return response.content.strip()

        except Exception as e:
            return f"Sorry, I couldn’t process that request through Azure OpenAI right now ({e}). Please try again later."

    # Default fallback for anything else
    return "That doesn’t look like a shipping-related question, but I’m MCS AI and I’m here to help! 😊 What would you like to know?"


def _get_current_consignee_code():
    """Get current consignee codes from thread-local storage"""
    return getattr(threading.current_thread(), 'consignee_code', None)

def _df_filtered_by_consignee(consignee_code=None):
    """Get DataFrame filtered by consignee codes if provided"""
    df = get_shipment_df()
    if consignee_code:
        # Extract just the numeric part from consignee codes
        numeric_codes = []
        for code in consignee_code:
            # Extract numeric part (e.g., "0045831" from "EDDIE BAUER LLC(0045831)")
            match = re.search(r'\((\d+)\)', code)
            if match:
                numeric_codes.append(match.group(1))
            else:
                # If already just numeric, use as is
                numeric_codes.append(code.strip())

        # Filter by consignee codes
        pattern = r"|".join([rf"\b{re.escape(code)}\b" for code in numeric_codes])
        mask = df['consignee_code_multiple'].astype(str).apply(
            lambda x: bool(re.search(pattern, x))
        )
        return df[mask]
    return df


def get_hot_upcoming_arrivals(query: str = None, consignee_code: str = None, **kwargs):
    """
    List ONLY hot containers arriving within a requested window.

    Robustness improvements:
    - Accepts 'question' kwarg as alias for query (agent sometimes calls tool with question=...)
    - Honors days from kwargs['days'], 'days=N' text, or 'next N days' text
      using an EXACT inclusive N-day window: [today .. today+(N-1)].
    - Optional transport_mode filter from query text (sea/air/road/rail/courier/sea-air).
    - Optional strict discharge_port/vehicle_arrival_lcn filter from query text per prompts policy.
    """

    # ---- 0) Normalize incoming text (Tool sometimes passes question=...) ----
    q = (query or kwargs.get("question") or kwargs.get("input") or "").strip()

    # 1) Determine requested days (tool override)
    requested_days = kwargs.get("days", None)

    if requested_days is None:
        m = re.search(r"\bdays\s*=\s*(\d{1,3})\b", q, flags=re.IGNORECASE)
        if m:
            requested_days = int(m.group(1))

    if requested_days is None:
        m = re.search(
            r"(?:next|upcoming|within|in\s+next|in\s+the\s+next|in)\s+(\d{1,3})\s+days?\b",
            q,
            flags=re.IGNORECASE,
        )
        if m:
            requested_days = int(m.group(1))

    # 2) Compute time window
    if requested_days is not None:
        n = max(int(requested_days), 1)
        today = pd.Timestamp.today().normalize()
        start_date = today
        end_date = today + pd.Timedelta(days=n - 1)
        period_desc = f"next {n} days (exact)"
    else:
        start_date, end_date, period_desc = parse_time_period(q)
        start_date = pd.Timestamp(start_date).normalize()
        end_date = pd.Timestamp(end_date).normalize()

    try:
        logger.info(f"[get_hot_upcoming_arrivals] Query: {q}")
        logger.info(f"[get_hot_upcoming_arrivals] Time window: {start_date}..{end_date} ({period_desc})")
    except Exception:
        pass

    # 3) Load data + optional consignee_code narrowing
    df = _df()
    if df is None or getattr(df, "empty", True):
        return []

    try:
        logger.info(f"[get_hot_upcoming_arrivals] Initial dataset: {len(df)} rows")
    except:
        pass

    if consignee_code and "consignee_code_multiple" in df.columns:
        codes = [c.strip() for c in str(consignee_code).split(",") if c.strip()]
        if codes:
            pattern = r"|".join([rf"\b{re.escape(code)}\b" for code in codes])
            df = df[
                df["consignee_code_multiple"]
                .astype(str)
                .apply(lambda x: bool(re.search(pattern, x, re.IGNORECASE)))
            ].copy()
        try:
            logger.info(f"[get_hot_upcoming_arrivals] After consignee filter ({consignee_code}): {len(df)} rows")
        except:
            pass

    if df.empty:
        return []

    # 3A) Transport mode filter (optional)
    try:
        modes = extract_transport_modes(q) or set()
    except Exception:
        modes = set()

    # **CRITICAL FIX**: Normalize modes -> dataset values (typically "SEA", "AIR", etc.)
    mode_map = {
        "sea": "SEA",
        "ocean": "SEA",
        "air": "AIR",
        "airfreight": "AIR",
        "road": "ROAD",
        "truck": "ROAD",
        "rail": "RAIL",
        "courier": "COURIER",
        "sea-air": "SEA-AIR",
    }
    wanted_modes = set()
    for m in modes:
        normalized = mode_map.get(m.lower(), str(m).upper())
        wanted_modes.add(normalized)

    if wanted_modes and "transport_mode" in df.columns:
        tm = df["transport_mode"].astype(str).str.upper()
        df = df[tm.apply(lambda s: any(wm in s for wm in wanted_modes))].copy()
        try:
            logger.info(f"[get_hot_upcoming_arrivals] After transport_mode filter ({wanted_modes}): {len(df)} rows")
        except:
            pass

    if df.empty:
        return []

    # 3B) Strict location filter (optional)
    # Policy: if code present -> require "(CODE)" in discharge_port or vehicle_arrival_lcn
    port_cols = [c for c in ["discharge_port", "vehicle_arrival_lcn"] if c in df.columns]

    def _extract_loc_code_or_name(text: str):
        # Prefer explicit "NAME(CODE)"
        m_paren = re.search(r"([A-Z0-9 ,\.\-']+?)\s*\(([A-Z0-9]{2,6})\)", text.upper())
        if m_paren:
            return m_paren.group(2).strip(), m_paren.group(1).strip()

        # Code-only token (3–6 alnum) heuristic - ONLY if query has location keywords
        location_keywords = r"\b(?:port|location|at|to|from|via|arriving|departure|destination|discharge)\b"
        has_location_context = bool(re.search(location_keywords, text, re.IGNORECASE))

        if has_location_context:
            tokens = re.findall(r"\b[A-Z0-9]{3,6}\b", text.upper())
            stop = {
                "NEXT", "DAYS", "DAY", "HOT", "ETA", "ATA", "ETD",
                "WITHIN", "UPCOMING", "ARRIVE", "ARRIVING", "ARRIVALS",
                "CONTAINER", "CONTAINERS", "FOR", "USER", "CONSIGNEE", "CODE",
                "SEA", "AIR", "ROAD", "RAIL", "COURIER",
                # Common query words that are NOT port codes
                "PLEASE", "WHAT", "ARE", "THE", "SHOW", "LIST", "GET", "GIVE",
                "TELL", "KNOW", "LET", "ME", "MY", "ALL", "ANY", "SOME",
                "FROM", "TO", "IN", "AT", "ON", "BY", "WITH", "THIS", "THAT",
                "WEEK", "MONTH", "YEAR", "TODAY", "NOW", "SOON", "LATER", "VIA",
                "PORT", "LOCATION", "DESTINATION", "DISCHARGE", "DEPARTURE"
            }
            for t in tokens:
                if t not in stop and not t.isdigit():
                    return t, None

            # Name heuristic: "at/to/in/from <proper name>" (only with location context)
            # More restrictive: name should be capitalized and reasonably port-like
            m_name = re.search(r"\b(?:at|to|from|via)\s+([A-Z][A-Za-z\s\.\-']{3,25})\b(?:\s+port|\s+location|$)", text, flags=re.IGNORECASE)
            if m_name:
                name_candidate = m_name.group(1).strip()
                # Exclude common time/direction words
                if name_candidate.upper() not in {"NEXT", "THIS", "THAT", "EVERY", "EACH"}:
                    return None, name_candidate

        return None, None

    loc_code, loc_name = _extract_loc_code_or_name(q)
    if port_cols and (loc_code or loc_name):
        try:
            logger.info(f"[get_hot_upcoming_arrivals] Location filter: code={loc_code}, name={loc_name}")
        except:
            pass

        if loc_code:
            code_pat = re.compile(rf"\({re.escape(loc_code)}\)", re.IGNORECASE)
            mask = pd.Series(False, index=df.index)
            for c in port_cols:
                mask |= df[c].astype(str).apply(lambda x: bool(code_pat.search(x)))
            df = df[mask].copy()
        else:
            name_pat = re.compile(re.escape(str(loc_name)), re.IGNORECASE)
            mask = pd.Series(False, index=df.index)
            for c in port_cols:
                mask |= df[c].astype(str).apply(lambda x: bool(name_pat.search(x)))
            df = df[mask].copy()

        try:
            logger.info(f"[get_hot_upcoming_arrivals] After location filter: {len(df)} rows")
        except:
            pass

    if df.empty:
        return []

    # 4) Hot-only filter
    hot_flag_cols = [c for c in df.columns if "hot_container_flag" in c.lower()]
    if not hot_flag_cols:
        hot_flag_cols = [c for c in df.columns if "hot_container" in c.lower()]
    if not hot_flag_cols:
        try:
            logger.warning(f"[get_hot_upcoming_arrivals] No hot_container_flag column found")
        except:
            pass
        return []

    hot_col = hot_flag_cols[0]

    def _is_hot(v) -> bool:
        if pd.isna(v):
            return False
        if isinstance(v, bool):
            return v is True
        s = str(v).strip().upper()
        return s in {"Y", "YES", "TRUE", "1", "HOT", "T"}

    df = df[df[hot_col].apply(_is_hot)].copy()
    try:
        logger.info(f"[get_hot_upcoming_arrivals] After hot_container_flag filter: {len(df)} rows")
    except:
        pass

    if df.empty:
        return []

    # 5) Dates + upcoming arrivals filter
    date_cols = [c for c in ["revised_eta", "eta_dp", "ata_dp"] if c in df.columns]
    if not date_cols:
        try:
            logger.warning(f"[get_hot_upcoming_arrivals] No date columns found")
        except:
            pass
        return []

    df = ensure_datetime(df, date_cols)

    # ETA preference: revised_eta > eta_dp
    if "revised_eta" in df.columns and "eta_dp" in df.columns:
        df["eta_for_filter"] = df["revised_eta"].combine_first(df["eta_dp"])
    elif "revised_eta" in df.columns:
        df["eta_for_filter"] = df["revised_eta"]
    else:
        df["eta_for_filter"] = df["eta_dp"]

    eta_norm = df["eta_for_filter"].dt.normalize()
    date_mask = df["eta_for_filter"].notna() & (eta_norm >= start_date) & (eta_norm <= end_date)

    # Upcoming only: exclude already arrived
    if "ata_dp" in df.columns:
        date_mask &= df["ata_dp"].isna()

    result = df[date_mask].copy()
    try:
        logger.info(f"[get_hot_upcoming_arrivals] After date/upcoming filter (start={start_date}, end={end_date}): {len(result)} rows")
    except:
        pass

    if result.empty:
        return []

    # Final guard (prevents leakage beyond window)
    eta_norm_r = result["eta_for_filter"].dt.normalize()
    result = result[result["eta_for_filter"].notna() & (eta_norm_r >= start_date) & (eta_norm_r <= end_date)].copy()
    if result.empty:
        return []

    # 6) Output shaping (no eta_for_filter in observation)
    result = result.sort_values("eta_for_filter", ascending=True).head(200).copy()

    out_cols = [
        "container_number",
        "po_number_multiple",
        "discharge_port",
        "revised_eta",
        "eta_dp",
        "consignee_code_multiple",
        "transport_mode",

    ]
    out_cols = [c for c in out_cols if c in result.columns]
    out_df = result[out_cols].copy()

    # Format date strings for clean JSON
    for dcol in ["revised_eta", "eta_dp"]:
        if dcol in out_df.columns and pd.api.types.is_datetime64_any_dtype(out_df[dcol]):
            out_df[dcol] = out_df[dcol].dt.strftime("%Y-%m-%d")

    try:
        logger.info(f"[get_hot_upcoming_arrivals] Returning {len(out_df)} hot container records")
    except:
        pass

    return out_df.where(pd.notnull(out_df), None).to_dict(orient="records")


def check_transit_status(query: str, consignee_code: str = None, **kwargs) -> str:
    """Question 14: Check if cargo/PO is currently in transit"""

    # Check if query is about transit time
    if isinstance(query, str) and "taking more than" in query.lower() and "days of transit" in query.lower():
        # Extract the number of days
        days_match = re.search(r"more than (\d+) days", query.lower())
        days = int(days_match.group(1)) if days_match else 10

        df = _df()  # Get dataframe with consignee filtering

        # Apply additional consignee filter if provided
        if consignee_code and "consignee_code_multiple" in df.columns:
            codes = [c.strip() for c in str(consignee_code).split(",") if c.strip()]
            mask = pd.Series(False, index=df.index)
            for c in codes:
                mask |= df["consignee_code_multiple"].astype(str).str.contains(re.escape(c), na=False)
            df = df[mask].copy()

        # Ensure datetime for all required columns
        df = ensure_datetime(df, ["eta_fd", "etd_lp", "atd_lp"])

        # Filter containers with eta_fd and either etd_lp or atd_lp
        valid_df = df[df["eta_fd"].notna() & ((df["etd_lp"].notna()) | (df["atd_lp"].notna()))].copy()

        # Calculate transit days using conditional logic:
        # If atd_lp is not null, use eta_fd - atd_lp
        # Otherwise, use eta_fd - etd_lp
        valid_df["transit_days"] = valid_df.apply(
            lambda row: (row["eta_fd"] - row["atd_lp"]).days if pd.notna(row["atd_lp"])
            else (row["eta_fd"] - row["etd_lp"]).days,
            axis=1
        )

        # Filter for transit_days >= days
        result_df = valid_df[valid_df["transit_days"] >= days].copy()

        # Prepare output
        if result_df.empty:
            return f"No containers found taking {days} or more days of transit time."

        cols = ["container_number", "etd_lp", "atd_lp", "eta_fd", "transit_days",
                "consignee_code_multiple", "discharge_port", "po_number_multiple"]
        cols = [c for c in cols if c in result_df.columns]

        out_df = result_df[cols].sort_values("transit_days", ascending=False).head(100).copy()

        # Format dates
        for col in ["etd_lp", "atd_lp", "eta_fd"]:
            if col in out_df.columns and pd.api.types.is_datetime64_any_dtype(out_df[col]):
                out_df[col] = out_df[col].dt.strftime("%Y-%m-%d")

        # Return as dictionary
        return out_df.where(pd.notnull(out_df), None).to_dict(orient="records")

    # Original PO transit check logic
    po_no = extract_po_number(query)
    if not po_no:
        return "Please specify a valid PO number."

    df = _df()  # Automatically filters by consignee
    po_col = "po_number_multiple" if "po_number_multiple" in df.columns else "po_number"
    rows = df[df[po_col].astype(str).str.contains(po_no, case=False, na=False)]

    if rows.empty:
        return f"No data found for PO {po_no} or you are not authorized to access this PO."

    row = rows.iloc[0]

    # Check if reached final destination or container returned
    fd_reached = pd.notnull(row.get('delivery_date_to_consignee'))
    container_returned = pd.notnull(row.get('empty_container_return_date'))
    departure_confirmed = pd.notnull(row.get('atd_lp'))

    if fd_reached or container_returned:
        return f"PO {po_no} has completed its journey."
    elif departure_confirmed:
        current_location = row.get('last_cy_location', 'In transit')
        return f"Yes, PO {po_no} is in transit. Current location: {current_location}"
    else:
        return f"PO {po_no} has not yet departed from load port."

def get_containers_by_carrier(query: str) -> str:
    """Questions 19-20: Containers handled/shipped by carrier"""

    # Extract carrier name and days
    carrier_match = re.search(r'carrier\s+([A-Z0-9\s]+)', query, re.IGNORECASE)
    days_match = re.search(r'(\d+)\s+days', query, re.IGNORECASE)

    if not carrier_match:
        return "Please specify a carrier name."

    carrier = carrier_match.group(1).strip()
    days = int(days_match.group(1)) if days_match else 30

    df = _df()  # Automatically filters by consignee
    df = ensure_datetime(df, ["atd_lp", "ata_dp"])

    # Filter by carrier
    carrier_mask = df['final_carrier_name'].astype(str).str.contains(carrier, case=False, na=False)

    # Date range
    today = pd.Timestamp.today().normalize()
    start_date = today - pd.Timedelta(days=days)

    if "ship" in query.lower():
        # Shipped (ATD)
        date_mask = (df['atd_lp'] >= start_date) & (df['atd_lp'] <= today)
        date_col = 'atd_lp'
        action = 'shipped'
    else:
        # Handled (any milestone)
        date_mask = (df['ata_dp'] >= start_date) & (df['ata_dp'] <= today)
        date_col = 'ata_dp'
        action = 'handled'

    result = df[carrier_mask & date_mask]

    if result.empty:
        return f"No containers {action} by {carrier} in the last {days} days for your authorized consignees."

    cols = ['container_number', 'final_carrier_name', date_col, 'discharge_port']
    result = result[cols].head(15)
    result[date_col] = result[date_col].dt.strftime("%Y-%m-%d")

    return f"Containers {action} by {carrier} in last {days} days:\n{result.to_string(index=False)}"

# Helper: parse supplier name (strip trailing "(code)")
def _parse_supplier_name(q: str) -> str:
    # try "supplier X..." or "from X..."
    m = re.search(r'(?:supplier|from)\s+([A-Z0-9&\.\'\-\s]+)', q, re.IGNORECASE)
    name = m.group(1) if m else ""
    # fallback: grab longest caps span
    if not name:
        caps = re.findall(r'[A-Z][A-Z0-9&\.\'\-\s]{5,}', q.upper())
        name = max(caps, key=len) if caps else ""
    name = name.strip()
    # remove trailing "(001234)" etc.
    name = re.sub(r'\s*\([^)]+\)\s*$', '', name).strip()
    return name


def get_supplier_in_transit(query: str) -> str:
    """
    Containers/POs from a supplier that are still in transit.
    Logic:
      - supplier_vendor_name contains supplier NAME (ignore code in parentheses)
      - ata_dp is null
      - empty_container_return_date is null (and if present, empty_container_return_lcn is null)
      - delivery_date_to_consignee is null
    Returns list[dict] with ETA preference revised_eta > eta_dp.
    """

    supplier = _parse_supplier_name(query)
    if not supplier:
        return "Please specify a supplier name."

    df = _df()
    if 'supplier_vendor_name' not in df.columns:
        return "Supplier vendor name column not found in the dataset."

    # supplier match (case-insensitive, ignore codes after '(')
    sup_mask = df['supplier_vendor_name'].astype(str).str.upper().str.contains(re.escape(supplier.upper()), na=False)

    # ensure dates we use
    parse_cols = [c for c in ['revised_eta', 'eta_dp', 'ata_dp',
                              'delivery_date_to_consignee', 'empty_container_return_date'] if c in df.columns]
    df = ensure_datetime(df, parse_cols)

    # helper: treat "", "nan", "nat", "none", "null" as null
    def _nullish(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(True, index=df.index)
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            return s.isna()
        s_str = s.astype(str).str.strip().str.upper()
        return s.isna() | s_str.isin({"", "NAN", "NAT", "NONE", "NULL"})

    # transit conditions
    not_arrived_dp = _nullish('ata_dp')          # must NOT have arrived at DP
    not_delivered = _nullish('delivery_date_to_consignee')
    not_returned_date = _nullish('empty_container_return_date')
    not_returned_loc = _nullish('empty_container_return_lcn')
    not_returned = not_returned_date & not_returned_loc

    subset = df[sup_mask & not_arrived_dp & not_delivered & not_returned].copy()
    if subset.empty:
        return f"No containers/POs from supplier '{supplier}' are still in transit for your authorized consignees."

    # ETA preference
    if 'revised_eta' in subset.columns and 'eta_dp' in subset.columns:
        subset['eta_for_filter'] = subset['revised_eta'].where(subset['revised_eta'].notna(), subset['eta_dp'])
    elif 'revised_eta' in subset.columns:
        subset['eta_for_filter'] = subset['revised_eta']
    else:
        subset['eta_for_filter'] = subset['eta_dp'] if 'eta_dp' in subset.columns else pd.NaT

    cols = [c for c in ['container_number', 'po_number_multiple', 'supplier_vendor_name',
                        'discharge_port', 'eta_for_filter', 'revised_eta', 'eta_dp'] if c in subset.columns]
    out = subset[cols]
    out = safe_sort_dataframe(out, 'eta_for_filter', ascending=True).head(100)

    # format dates
    for d in ['eta_for_filter', 'revised_eta', 'eta_dp']:
        if d in out.columns and pd.api.types.is_datetime64_any_dtype(out[d]):
            out[d] = out[d].dt.strftime('%Y-%m-%d')

    out = out.rename(columns={'eta_for_filter': 'eta'})
    return out.where(pd.notnull(out), None).to_dict(orient='records')


def ensure_datetime(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Convert specified columns to datetime using explicit formats to avoid inference warnings."""
    known_formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%m/%d/%Y %H:%M",
        "%d/%m/%Y %H:%M",
        "%d-%m-%Y",
        "%d-%m-%Y %H:%M:%S",
        "%d-%b-%Y",
        "%d-%b-%Y %H:%M:%S",
        "%m/%d/%Y %I:%M:%S %p",  # e.g., 2/20/2025 12:00:00 AM
    ]
    for col in columns:
        if col not in df.columns:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        s = df[col].astype(str).str.strip().replace({"": None, "NaN": None, "nan": None, "NaT": None})
        parsed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        for fmt in known_formats:
            mask = parsed.isna()
            if not mask.any():
                break
            try:
                parsed.loc[mask] = pd.to_datetime(s[mask], format=fmt, errors="coerce")
            except Exception:
                continue
        df[col] = parsed
    return df

def get_supplier_last_days(query: str) -> str:
    """
    Containers from a supplier in the last N days (arrived).
    Logic:
      - supplier_vendor_name contains supplier NAME
      - ata_dp not null and within [today-N, today]; default N=30
    """

    supplier = _parse_supplier_name(query)
    if not supplier:
        return "Please specify a supplier name."
    m = re.search(r'(?:last|past)\s+(\d{1,3})\s+days?', query, re.IGNORECASE)
    days = int(m.group(1)) if m else 30

    df = _df()
    if 'supplier_vendor_name' not in df.columns:
        return "Supplier vendor name column not found in the dataset."

    sup_mask = df['supplier_vendor_name'].astype(str).str.upper().str.contains(re.escape(supplier.upper()), na=False)
    df = ensure_datetime(df, ['ata_dp'])
    if 'ata_dp' not in df.columns:
        return "ATA column (ata_dp) not found."

    today = pd.Timestamp.today().normalize()
    start = today - pd.Timedelta(days=days)

    mask = sup_mask & df['ata_dp'].notna() & (df['ata_dp'] >= start) & (df['ata_dp'] <= today)
    subset = df[mask].copy()
    if subset.empty:
        return f"No containers from supplier '{supplier}' in the last {days} days for your authorized consignees."

    cols = [c for c in ['container_number', 'po_number_multiple', 'supplier_vendor_name',
                        'discharge_port', 'ata_dp'] if c in subset.columns]
    out = subset[cols]
    out = safe_sort_dataframe(out, 'ata_dp', ascending=False).head(150)
    if 'ata_dp' in out.columns and pd.api.types.is_datetime64_any_dtype(out['ata_dp']):
        out['ata_dp'] = out['ata_dp'].dt.strftime('%Y-%m-%d')
    return out.where(pd.notnull(out), None).to_dict(orient='records')


def get_containers_PO_OBL_by_supplier(query: str) -> str:
    """
    Handle supplier/shipper-related queries including:
    1. Lookup supplier for a specific container/PO/OBL
    2. List containers arriving from a supplier (upcoming/delayed)
    3. Filter by supplier with date windows and delay thresholds.
    """
    query = (query or "").strip()
    q_lower = query.lower()
    q_upper = query.upper()

    df = _df()  # already consignee-filtered via thread context

    if "supplier_vendor_name" not in df.columns:
        return "Supplier/vendor information not available in the dataset."

    # ---------- explicit identifiers ----------
    raw_container = extract_container_number(query)
    raw_po = extract_po_number(query)
    try:
        raw_obl = extract_ocean_bl_number(query)
    except Exception:
        raw_obl = None

    # Fallback: if query is just an alphanumeric string (8-20 chars), treat as OBL if not PO/Container
    if not raw_obl and not raw_po and not raw_container:
         # Check if query is a single token or "OBL <token>"
         cleaned_q = query.strip().upper()
         # Case 1: Just the token
         if re.match(r'^[A-Z0-9]{8,20}$', cleaned_q):
             raw_obl = cleaned_q
         # Case 2: "OBL <token>"
         else:
             m_obl = re.search(r'\bOBL\s+([A-Z0-9]{8,20})\b', cleaned_q)
             if m_obl:
                 raw_obl = m_obl.group(1)

    # Did user explicitly say PO / purchase order?
    mentions_po = bool(re.search(r'\b(po|purchase\s+order)\b', q_lower))

    # If the user explicitly says PO, we should **not** treat a pure number as container
    container_no = raw_container if (raw_container and not mentions_po) else None
    po_no = raw_po
    obl_no = raw_obl

    # Log what we extracted for debugging
    try:
        logger.info(f"[get_containers_PO_OBL_by_supplier] Extracted: container={container_no}, po={po_no}, obl={obl_no}")
    except:
        pass

    # CASE 1 — Direct lookup: identifier present OR "supplier for"
    supplier_lookup_phrases = [
        "what is the supplier",
        "what is the shipper",
        "who is the supplier",
        "who is the shipper",
        "supplier for",
        "shipper for",
        "show supplier",
        "show shipper",
        "tell me the supplier",
        "tell me the shipper",
        "get supplier",
        "get shipper",
        "find supplier",
        "find shipper",
    ]
    is_lookup_query = any(p in q_lower for p in supplier_lookup_phrases)

    # If we have any identifier, treat as lookup
    if container_no or po_no or obl_no:
        is_lookup_query = True

    if is_lookup_query:
        # 1) PO → supplier (PREFER when query talks about PO)
        if po_no:
            po_norm = _normalize_po_token(po_no)
            po_col = (
                "po_number_multiple"
                if "po_number_multiple" in df.columns
                else ("po_number" if "po_number" in df.columns else None)
            )
            if not po_col:
                return "PO column not found in the dataset."

            mask = df[po_col].apply(lambda cell: _po_in_cell(cell, po_norm))
            rows = df[mask].copy()

            try:
                logger.info(f"[get_containers_PO_OBL_by_supplier] PO lookup: po_norm={po_norm}, found {len(rows)} rows")
            except:
                pass

            if rows.empty:
                return f"No data found for PO {po_no}."

            suppliers = (
                rows["supplier_vendor_name"]
                .dropna()
                .astype(str)
                .map(str.strip)
            )
            suppliers = [
                s
                for s in suppliers.unique().tolist()
                if s and s.upper() not in {"NAN", "NONE", "NULL"}
            ]

            containers = (
                rows["container_number"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
                if "container_number" in rows.columns
                else []
            )

            return [
                {
                    "po_number": po_no,
                    "supplier_vendor_name": ", ".join(suppliers)
                    if suppliers
                    else "Not Available",
                    "container_count": len(containers),
                    "containers": ", ".join(containers[:5])
                    + ("..." if len(containers) > 5 else ""),
                    "discharge_port": rows.iloc[0].get("discharge_port"),
                    "consignee_code_multiple": rows.iloc[0].get(
                        "consignee_code_multiple"
                    ),
                }
            ]

        # 2) Container → supplier
        if container_no:
            clean = clean_container_number(container_no)
            df_loc = df.copy()
            if "container_number" not in df_loc.columns:
                return f"No data found for container {container_no}."

            df_loc["container_number"] = (
                df_loc["container_number"]
                .astype(str)
                .str.upper()
                .str.replace(r"[^A-Z0-9]", "", regex=True)
            )

            rows = df_loc[df_loc["container_number"] == clean]
            if rows.empty:
                rows = df_loc[
                    df_loc["container_number"].astype(str).str.contains(
                        clean, case=False, na=False
                    )
                ]

            try:
                logger.info(f"[get_containers_PO_OBL_by_supplier] Container lookup: clean={clean}, found {len(rows)} rows")
            except:
                pass

            if rows.empty:
                return f"No data found for container {container_no}."

            row = rows.iloc[0]
            supplier = row.get("supplier_vendor_name")
            if pd.isna(supplier) or str(supplier).strip().upper() in {
                "",
                "NAN",
                "NONE",
                "NULL",
            }:
                supplier = "Not Available"

            return [
                {
                    "container_number": row.get("container_number"),
                    "supplier_vendor_name": supplier,
                    "po_number_multiple": row.get("po_number_multiple"),
                    "ocean_bl_no_multiple": row.get("ocean_bl_no_multiple"),
                    "discharge_port": row.get("discharge_port"),
                    "eta_dp": row.get("eta_dp"),
                    "ata_dp": row.get("ata_dp"),
                    "consignee_code_multiple": row.get("consignee_code_multiple"),
                }
            ]

        # 3) OBL/BL → supplier (FIXED VERSION)
        if obl_no:
            # Use the robust helper to find the BL column
            bl_col = _find_ocean_bl_col(df)

            try:
                logger.info(f"[get_containers_PO_OBL_by_supplier] OBL lookup: obl_no={obl_no}, bl_col={bl_col}")
            except:
                pass

            if not bl_col:
                return "No ocean BL column found in the dataset."

            # Normalize the OBL number for matching
            bl_norm = _normalize_bl_token(obl_no)

            try:
                logger.info(f"[get_containers_PO_OBL_by_supplier] Normalized OBL: {bl_norm}")
            except:
                pass

            # Use the robust cell matching function
            mask = df[bl_col].apply(lambda cell: _bl_in_cell(cell, bl_norm))
            rows = df[mask].copy()

            try:
                logger.info(f"[get_containers_PO_OBL_by_supplier] OBL mask found {len(rows)} rows")
                if len(rows) > 0:
                    logger.info(f"[get_containers_PO_OBL_by_supplier] Sample BL values: {rows[bl_col].head().tolist()}")
            except:
                pass

            if rows.empty:
                return f"No data found for OBL {obl_no}."

            suppliers = (
                rows["supplier_vendor_name"]
                .dropna()
                .astype(str)
                .map(str.strip)
            )
            suppliers = [
                s
                for s in suppliers.unique().tolist()
                if s and s.upper() not in {"NAN", "NONE", "NULL"}
            ]

            containers = (
                rows["container_number"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
                if "container_number" in rows.columns
                else []
            )

            # **CRITICAL FIX**: Add early return here to prevent falling through to CASE 2/3
            return [
                {
                    "ocean_bl_no": obl_no,
                    "supplier_vendor_name": ", ".join(suppliers)
                    if suppliers
                    else "Not Available",
                    "container_count": len(containers),
                    "containers": ", ".join(containers[:5])
                    + ("..." if len(containers) > 5 else ""),
                    "discharge_port": rows.iloc[0].get("discharge_port"),
                    "consignee_code_multiple": rows.iloc[0].get(
                        "consignee_code_multiple"
                    ),
                }
            ]

        # If we reach here in lookup mode but no identifier matched
        return "Please specify a valid container number, PO number, or OBL number to look up the supplier."

    # CASE 2/3 — supplier-name, delayed/upcoming logic continues below...
    supplier_name = None

    m = re.search(
        r"(?:from|by)\s+(?:supplier|shipper|vendor)\s+([A-Z0-9\(\)&\.\'\-\s]{3,})",
        q_upper,
        re.IGNORECASE,
    )
    if m:
        supplier_name = m.group(1).strip()

    if not supplier_name:
        m = re.search(
            r"(?:supplier|shipper|vendor)\s+([A-Z0-9\(\)&\.\'\-\s]{3,})",
            q_upper,
            re.IGNORECASE,
        )
        if m:
            candidate = m.group(1).strip()
            if not any(
                word in candidate.upper()
                for word in [
                    "LATE",
                    "DELAYED",
                    "ARRIVING",
                    "NEXT",
                    "LAST",
                    "DAYS",
                    "FOR",
                ]
            ):
                supplier_name = candidate

    if supplier_name:
        supplier_name = re.sub(r"\s*\([^)]+\)\s*$", "", supplier_name).strip()
        supplier_name = re.sub(
            r"\s+(LATE|DELAYED|ARRIVING|IN\s+NEXT|LAST|WITHIN).*?$",
            "",
            supplier_name,
            flags=re.IGNORECASE,
        ).strip()

    if supplier_name and len(supplier_name) > 2:
        sup_mask = df["supplier_vendor_name"].astype(str).str.upper().str.contains(
            re.escape(supplier_name.upper()), na=False
        )
        df = df[sup_mask].copy()
        if df.empty:
            return f"No containers found from supplier '{supplier_name}' for your authorized consignees."

    is_delayed_query = any(
        w in q_lower for w in ["delay", "late", "overdue", "behind", "missed"]
    )
    is_upcoming_query = any(
        w in q_lower
        for w in [
            "upcoming",
            "arriving",
            "arrive",
            "next",
            "coming",
            "expected",
            "will arrive",
        ]
    )
    is_transit_query = any(
        p in q_lower for p in ["in transit", "still moving", "not arrived", "on the way"]
    )
    is_arrived_query = any(
        w in q_lower for w in ["arrived", "reached", "landed", "delivered"]
    ) and not is_upcoming_query

    # Check if we actually applied any filter
    has_supplier_filter = (supplier_name is not None and len(supplier_name) > 2)
    has_status_filter = (is_delayed_query or is_upcoming_query or is_transit_query or is_arrived_query)

    if not has_supplier_filter and not has_status_filter:
         return "Please specify a supplier, container, PO, or OBL number."

    start_date, end_date, period_desc = parse_time_period(query)

    date_cols = [
        "eta_dp",
        "revised_eta",
        "ata_dp",
        "atd_lp",
        "etd_lp",
        "delivery_date_to_consignee",
        "empty_container_return_date",
    ]
    df = ensure_datetime(df, [c for c in date_cols if c in df.columns])

    if is_delayed_query:
        if "ata_dp" not in df.columns or "eta_dp" not in df.columns:
            return "Required columns for delay calculation not found."

        arrived_mask = df["ata_dp"].notna() & df["eta_dp"].notna()
        df_arrived = df[arrived_mask].copy()
        if df_arrived.empty:
            ctx = f" from supplier '{supplier_name}'" if supplier_name else ""
            return f"No arrived containers found{ctx}."

        df_arrived["delay_days"] = (
            (df_arrived["ata_dp"] - df_arrived["eta_dp"]).dt.total_seconds() / 86400
        ).round().astype(int)

        delay_threshold = None
        delay_operator = ">"
        patterns = [
            (r"(?:less\s+than|under|below|<)\s*(\d+)\s+days?", "<"),
            (r"(?:more\s+than|over|above|>)\s*(\d+)\s+days?", ">"),
            (r"(?:at\s+least|minimum|>=)\s*(\d+)\s+days?", ">="),
            (r"(?:up\s+to|maximum|<=)\s*(\d+)\s+days?", "<="),
            (r"(?:exactly|equal\s+to|=)\s*(\d+)\s+days?", "=="),
        ]
        for pat, op in patterns:
            m = re.search(pat, query, re.IGNORECASE)
            if m:
                delay_threshold = int(m.group(1))
                delay_operator = op
                break

        if delay_threshold is not None:
            if delay_operator == "<":
                delay_mask = (df_arrived["delay_days"] > 0) & (
                    df_arrived["delay_days"] < delay_threshold
                )
            elif delay_operator == ">":
                delay_mask = df_arrived["delay_days"] > delay_threshold
            elif delay_operator == ">=":
                delay_mask = df_arrived["delay_days"] >= delay_threshold
            elif delay_operator == "==":
                delay_mask = df_arrived["delay_days"] == delay_threshold
            elif delay_operator == "<=":
                delay_mask = (df_arrived["delay_days"] > 0) & (
                    df_arrived["delay_days"] <= delay_threshold
                )
        else:
            delay_mask = df_arrived["delay_days"] > 0

        result = df_arrived[delay_mask].copy()
        if result.empty:
            ctx = f" from supplier '{supplier_name}'" if supplier_name else ""
            if delay_threshold is not None:
                return f"No containers with delay {delay_operator} {delay_threshold} days{ctx}."
            return f"No delayed containers found{ctx}."

        result = safe_sort_dataframe(result, "delay_days", ascending=False)
        out_cols = [
            c
            for c in [
                "container_number",
                "supplier_vendor_name",
                "po_number_multiple",
                "eta_dp",
                "ata_dp",
                "delay_days",
                "discharge_port",
                "consignee_code_multiple",
            ]
            if c in result.columns
        ]
        out = result[out_cols].head(200).copy()
        for dcol in ["eta_dp", "ata_dp"]:
            if dcol in out.columns and pd.api.types.is_datetime64_any_dtype(out[dcol]):
                out[dcol] = out[dcol].dt.strftime("%Y-%m-%d")
        return out.where(pd.notnull(out), None).to_dict(orient="records")

    if df.empty:
        ctx = f" from supplier '{supplier_name}'" if supplier_name else ""
        return f"No containers found{ctx}."

    sort_col = (
        "eta_dp"
        if "eta_dp" in df.columns
        else ("ata_dp" if "ata_dp" in df.columns else None)
    )
    if sort_col:
        df = safe_sort_dataframe(df, sort_col, ascending=False)

    out_cols = [
        c
        for c in [
            "container_number",
            "supplier_vendor_name",
            "po_number_multiple",
            "ocean_bl_no_multiple",
            "discharge_port",
            "eta_dp",
            "ata_dp",
            "consignee_code_multiple",
        ]
        if c in df.columns
    ]
    out = df[out_cols].head(200).copy()
    for dcol in ["eta_dp", "ata_dp"]:
        if dcol in out.columns and pd.api.types.is_datetime64_any_dtype(out[dcol]):
            out[dcol] = out[dcol].dt.strftime("%Y-%m-%d")
    return out.where(pd.notnull(out), None).to_dict(orient="records")


# --- replace the later _normalize_po_token / _po_in_cell definitions with this robust version ---
def _normalize_po_token(s: str) -> str:
    """Normalize a PO token for comparison: strip, upper, keep alphanumerics."""
    if s is None:
        return ""
    s = str(s).strip().upper()
    return re.sub(r'[^A-Z0-9]', '', s)

def _po_in_cell(cell: str, po_norm: str) -> bool:
    """
    Return True if normalized PO exists in a comma/sep-separated cell.
    Robust to tokens like 'PO5302816722' vs '5302816722' and multi-valued cells.
    """
    if pd.isna(cell) or not po_norm:
        return False
    parts = re.split(r'[,;/\|\s]+', str(cell))
    q_digits = po_norm.isdigit()
    q_strip_po = po_norm[2:] if po_norm.startswith('PO') else po_norm
    for p in parts:
        if not p:
            continue
        pn = _normalize_po_token(p)
        if not pn:
            continue
        # exact match
        if pn == po_norm:
            return True
        # allow PO prefix/no prefix equivalence
        if q_digits and pn.endswith(q_strip_po):
            # accept PN like PO########## or just ##########
            if pn == q_strip_po or pn == f"PO{q_strip_po}":
                return True
        if not q_digits and po_norm.startswith("PO") and pn == q_strip_po:
            return True
    return False


def check_po_month_arrival(query: str) -> str:
    """
    Can PO arrive at destination by end of this month?

    Logic:
    1) If any row for PO has ata_dp NOT NULL -> it's already arrived. Return that.
    2) Otherwise, compute next_eta_fd = NVL(predictive_eta_fd, revised_eta_fd, eta_fd)
       and check if next_eta_fd <= last day of current month.
    """
    # --- robust PO extraction ---

    po_no = extract_po_number(query)
    if not po_no:
        m = re.search(r'(?:po(?:\s*number)?\s*[:#-]?\s*)?([A-Z0-9]{6,20})', query, re.IGNORECASE)
        po_no = m.group(1) if m else None
    if po_no and po_no.upper().startswith('PO') and po_no[2:].isdigit():
        po_no = po_no[2:]
    if not po_no:
        return "Please specify a valid PO number."

    po_norm = _normalize_po_token(po_no)
    df = _df()

    # choose PO column
    po_col = "po_number_multiple" if "po_number_multiple" in df.columns else ("po_number" if "po_number" in df.columns else None)
    if not po_col:
        return "PO column not found in the dataset."

    # match rows where normalized PO token is present in the multi-value cell
    mask = df[po_col].apply(lambda cell: _po_in_cell(cell, po_norm))
    matches = df[mask].copy()
    if matches.empty:
        return f"No data found for PO {po_no}."

    # ensure datetime for required fields
    date_cols = [c for c in ["ata_dp", "predictive_eta_fd", "revised_eta_fd", "eta_fd"] if c in matches.columns]
    if date_cols:
        matches = ensure_datetime(matches, date_cols)

    # last day of current month
    today = pd.Timestamp.today().normalize()
    first = pd.Timestamp(today.year, today.month, 1)
    last_day = first + pd.DateOffset(months=1) - pd.Timedelta(days=1)

    # 1) If any ata_dp exists -> already arrived
    if "ata_dp" in matches.columns and matches["ata_dp"].notna().any():
        arrived_rows = matches[matches["ata_dp"].notna()].copy()
        # pick earliest or show min as the arrival confirmation
        ata_min = arrived_rows["ata_dp"].min()
        if pd.notna(ata_min):
            dt_str = ata_min.strftime("%Y-%m-%d") if hasattr(ata_min, "strftime") else str(ata_min)
            return f"Yes, PO {po_no} has already arrived on {dt_str}."
        return f"Yes, PO {po_no} has already arrived."

    # 2) Not arrived yet -> use NVL(predictive_eta_fd, revised_eta_fd, eta_fd)
    # build next_eta_fd
    next_eta = pd.Series(pd.NaT, index=matches.index, dtype="datetime64[ns]")
    for c in ["predictive_eta_fd", "revised_eta_fd", "eta_fd"]:
        if c in matches.columns:
            next_eta = next_eta.fillna(matches[c])
    matches["_next_eta_fd"] = next_eta

    pending = matches[matches["_next_eta_fd"].notna()].copy()
    if pending.empty:
        return f"No ETA FD available for PO {po_no}."

    within = pending["_next_eta_fd"] <= last_day
    if within.any():
        eta_pick = pending.loc[within, "_next_eta_fd"].min()
        eta_str = eta_pick.strftime("%Y-%m-%d") if hasattr(eta_pick, "strftime") else str(eta_pick)
        # include containers if present
        conts = pending.loc[within, "container_number"].dropna().astype(str).unique().tolist() if "container_number" in pending.columns else []
        cont_str = f" Containers: {', '.join(conts)}." if conts else ""
        return f"Yes, PO {po_no} can arrive by {eta_str} (on or before month end {last_day.strftime('%Y-%m-%d')}).{cont_str}"
    else:
        eta_pick = pending["_next_eta_fd"].min()
        eta_str = eta_pick.strftime("%Y-%m-%d") if hasattr(eta_pick, "strftime") else str(eta_pick)
        conts = pending.loc[pending["_next_eta_fd"] == eta_pick, "container_number"].dropna().astype(str).unique().tolist() if "container_number" in pending.columns else []
        cont_str = f" Containers: {', '.join(conts)}." if conts else ""
        return f"No, PO {po_no} is expected on {eta_str} (after month end {last_day.strftime('%Y-%m-%d')}).{cont_str}"

#    ...existing code...

def get_weekly_status_changes(query: str) -> str:
    """Question 27: Weekly status changes"""
    df = _df()  # Automatically filters by consignee

    # Determine if current week or last week
    is_last_week = "last week" in query.lower()

    today = pd.Timestamp.today()
    if is_last_week:
        week_start = today - pd.Timedelta(days=today.weekday() + 7)
        week_end = week_start + pd.Timedelta(days=6)
        period = "last week"
    else:
        week_start = today - pd.Timedelta(days=today.weekday())
        week_end = today
        period = "this week"

    # Check all milestone date columns for changes in the week
    milestone_cols = ['ata_dp', 'delivery_date_to_consignee', 'empty_container_return_date', 'out_gate_at_last_cy']

    status_changes = []
    for col in milestone_cols:
        if col in df.columns:
            df_col = ensure_datetime(df, [col])
            week_updates = df_col[(df_col[col] >= week_start) & (df_col[col] <= week_end)]

            if not week_updates.empty:
                for _, row in week_updates.iterrows():
                    status_changes.append({
                        'container_number': row['container_number'],
                        'milestone': col.replace('_', ' ').title(),
                        'date': row[col].strftime('%Y-%m-%d'),
                        'location': row.get('discharge_port', '')
                    })

    if not status_changes:
        return f"No container status changes for {period} for your authorized consignees."

    # Format results
    result_lines = [f"Container status changes for {period}:"]
    for change in status_changes[:200]:  # Limit to 20 results
        result_lines.append(f"- {change['container_number']}: {change['milestone']} on {change['date']} at {change['location']}")

    return "\n".join(result_lines)

def get_current_location(query: str) -> str:
    """Question 30: Current container location"""
    container_no = extract_container_number(query)
    if not container_no:
        return "Please specify a valid container number."

    df = _df()  # Automatically filters by consignee
    rows = df[df["container_number"].astype(str).str.contains(container_no, case=False, na=False)]

    if rows.empty:
        return f"No data found for container {container_no} or you are not authorized to access this container."

    row = rows.iloc[0]

    # Determine current location based on latest milestone
    if pd.notnull(row.get('empty_container_return_date')):
        location = row.get('empty_container_return_lcn', 'Container depot')
        return f"Container {container_no} has been returned to {location}"
    elif pd.notnull(row.get('delivery_date_to_consignee')):
        location = row.get('delivery_location_to_consignee', 'Final destination')
        return f"Container {container_no} has been delivered to {location}"
    elif pd.notnull(row.get('out_gate_at_last_cy')):
        location = row.get('out_gate_at_last_cy_lcn', 'Last container yard')
        return f"Container {container_no} has departed from {location} and is en route to final destination"
    elif pd.notnull(row.get('equipment_arrived_at_last_cy')):
        location = row.get('equipment_arrival_at_last_lcn', 'Container yard')
        return f"Container {container_no} is currently at {location}"
    elif pd.notnull(row.get('ata_dp')):
        location = row.get('discharge_port', 'Discharge port')
        return f"Container {container_no} has arrived at {location} discharge port"
    elif pd.notnull(row.get('atd_lp')):
        return f"Container {container_no} is on the water, en route to {row.get('discharge_port', 'discharge port')}"
    else:
        return f"Container {container_no} status: Preparing for shipment at {row.get('load_port', 'load port')}"
# 1️⃣ Container Milestones

def get_container_milestones(input_str: str) -> str:
    """
    Retrieve all milestone events for a given container, PO, or OBL number.
    Search order:
      1. container_number
      2. po_number_multiple (ENHANCED)
      3. ocean_bl_no_multiple
    """

    query = str(input_str).strip()
    if not query:
        return "Please provide a container number, PO number, or OBL number."

    df = _df().copy()

    # Normalize required columns
    for col in ["container_number", "po_number_multiple", "ocean_bl_no_multiple"]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("").str.strip()

    container_no = None
    header_text = ""

    # 1) Try direct container match
    match_container = df[df["container_number"].str.replace(" ", "").str.upper() == query.replace(" ", "").upper()]
    if not match_container.empty:
        container_no = match_container.iloc[0]["container_number"]
        header_text = ""
        row = match_container.iloc[0]
    else:
        # 2) **ENHANCED PO MATCH** - Extract and normalize PO number
        po_no = extract_po_number(query)

        # Additional fallback for pure numeric PO (e.g., "5300009636")
        if not po_no:
            # Try to extract pure numeric sequence
            m = re.search(r'\b(\d{6,})\b', query)
            if m:
                po_no = m.group(1)

        if po_no:
            try:
                logger.info(f"[get_container_milestones] Searching for PO: {po_no}")
            except:
                pass

            # **CRITICAL FIX**: Use robust PO matching with normalization
            po_norm = _normalize_po_token(po_no)

            # Check if po_number_multiple column exists
            if "po_number_multiple" in df.columns:
                # Use the robust _po_in_cell matcher
                match_po = df[df["po_number_multiple"].apply(lambda cell: _po_in_cell(cell, po_norm))]

                try:
                    logger.info(f"[get_container_milestones] PO match results: {len(match_po)} rows found")
                except:
                    pass

                if not match_po.empty:
                    container_no = match_po.iloc[0]["container_number"]
                    header_text = f"The Container <con>{container_no}</con> is associated with the PO <po>{po_no}</po> . Status is in below : \n\n"
                    row = match_po.iloc[0]
                else:
                    # **FALLBACK**: Try simple contains match (less strict)
                    match_po_fallback = df[df["po_number_multiple"].str.upper().str.contains(po_no.upper(), na=False)]
                    if not match_po_fallback.empty:
                        container_no = match_po_fallback.iloc[0]["container_number"]
                        header_text = f"The Container <con>{container_no}</con> is associated with the PO <po>{po_no}</po> . Status is in below : \n\n"
                        row = match_po_fallback.iloc[0]
                    else:
                        return f"No record found for PO {po_no}."
            else:
                return "PO column (po_number_multiple) not found in dataset."
        else:
            # 3) Try OBL match
            match_obl = df[df["ocean_bl_no_multiple"].str.contains(query, case=False, na=False)]
            if not match_obl.empty:
                container_no = match_obl.iloc[0]["container_number"]
                header_text = f"The Container <con>{container_no}</con> is associated with the OBL <obl>{query}</obl> . Status is in below : \n\n"
                row = match_obl.iloc[0]
            else:
                return f"No record found for {query}."

    # ---- milestone rows with priority (prevents bad data ordering) ----
    milestone_defs = [
        ("<strong>Departed From</strong>", row.get("load_port"), row.get("atd_lp"), 20),
        ("<strong>Arrived at Final Load Port</strong>", row.get("final_load_port"), row.get("ata_flp"), 30),
        ("<strong>Departed from Final Load Port</strong>", row.get("final_load_port"), row.get("atd_flp"), 40),
        ("<strong>Expected at Discharge Port</strong>", row.get("discharge_port"), row.get("derived_ata_dp") or row.get("eta_dp"), 50),
        ("<strong>Reached at Discharge Port</strong>", row.get("discharge_port"), row.get("ata_dp"), 60),
        ("<strong>Reached at Last CY</strong>", row.get("last_cy_location"), row.get("equipment_arrived_at_last_cy"), 70),
        ("<strong>Out Gate at Last CY</strong>", row.get("out_gate_at_last_cy_lcn"), row.get("out_gate_at_last_cy"), 80),
        ("<strong>Delivered at</strong>", row.get("delivery_date_to_consignee_lcn"), row.get("delivery_date_to_consignee"), 90),
        ("<strong>Empty Container Returned to</strong>", row.get("empty_container_return_lcn"), row.get("empty_container_return_date"), 100),
    ]

    milestone_rows = []
    for event, location, raw_date, rank in milestone_defs:
        dt = pd.to_datetime(raw_date, errors="coerce")
        if pd.isna(dt):
            continue
        milestone_rows.append(
            {
                "event": event,
                "location": None if pd.isna(location) else location,
                "date": dt.strftime("%Y-%m-%d"),
                "_dt": dt,
                "_rank": rank,
            }
        )

    if not milestone_rows:
        return f"No milestones found for container {container_no}."

    milestones_df = pd.DataFrame(milestone_rows)

    # Pick "latest status" by (rank first, then date)
    last_row = max(milestone_rows, key=lambda x: (x["_rank"], x["_dt"]))
    latest_text = f"The Container <con>{container_no}</con> {last_row['event']} {last_row['location']} on {last_row['date']}"

    # Convert milestone dataframe to string (no internal helper cols)
    milestone_text = milestones_df[["event", "location", "date"]].to_string(index=False, header=False)

    result = (
        f"{header_text}"
        f"{latest_text}\n\n"
        f" <MILESTONE> {milestone_text}."
    )
    return result

# def get_container_milestones(input_str: str) -> str:
#     """
#     Retrieve all milestone events for a given container, PO, or OBL number.
#     Search order:
#       1. container_number
#       2. po_number_multiple
#       3. ocean_bl_no_multiple
#     Output:
#       Returns a descriptive text block exactly in your desired format.
#     """

#     query = str(input_str).strip()
#     if not query:
#         return "Please provide a container number, PO number, or OBL number."

#     df = _df().copy()

#     # Normalize required columns
#     for col in ["container_number", "po_number_multiple", "ocean_bl_no_multiple"]:
#         if col in df.columns:
#             df[col] = df[col].astype(str).fillna("").str.strip()

#     container_no = None
#     header_text = ""

#     # 1) Try direct container match
#     match_container = df[df["container_number"].str.replace(" ", "").str.upper() == query.replace(" ", "").upper()]
#     if not match_container.empty:
#         container_no = match_container.iloc[0]["container_number"]
#         header_text = ""
#         row = match_container.iloc[0]
#     else:
#         # 2) Try PO match
#         match_po = df[df["po_number_multiple"].str.contains(query, case=False, na=False)]
#         if not match_po.empty:
#             container_no = match_po.iloc[0]["container_number"]
#             header_text = f"The Container <con>{container_no}</con> is associated with the PO <po>{query}</po> . Status is in below : \n\n"
#             row = match_po.iloc[0]
#         else:
#             # 3) Try OBL match
#             match_obl = df[df["ocean_bl_no_multiple"].str.contains(query, case=False, na=False)]
#             if not match_obl.empty:
#                 container_no = match_obl.iloc[0]["container_number"]
#                 header_text = f"The Container <con>{container_no}</con> is associated with the OBL <obl>{query}</obl> . Status is in below : \n\n"
#                 row = match_obl.iloc[0]
#             else:
#                 return f"No record found for {query}."

#     # ---- milestone rows with priority (prevents bad data ordering from choosing wrong "latest") ----
#     # Higher rank = more final/completed status.
#     milestone_defs = [

#         ("<strong>Departed From</strong>", row.get("load_port"), row.get("atd_lp"), 20),
#         ("<strong>Arrived at Final Load Port</strong>", row.get("final_load_port"), row.get("ata_flp"), 30),
#         ("<strong>Departed from Final Load Port</strong>", row.get("final_load_port"), row.get("atd_flp"), 40),
#         ("<strong>Expected at Discharge Port</strong>", row.get("discharge_port"), row.get("derived_ata_dp") or row.get("eta_dp"), 50),
#         ("<strong>Reached at Discharge Port</strong>", row.get("discharge_port"), row.get("ata_dp"), 60),
#         ("<strong>Reached at Last CY</strong>", row.get("last_cy_location"), row.get("equipment_arrived_at_last_cy"), 70),
#         ("<strong>Out Gate at Last CY</strong>", row.get("out_gate_at_last_cy_lcn"), row.get("out_gate_at_last_cy"), 80),
#         ("<strong>Delivered at</strong>", row.get("delivery_date_to_consignee_lcn"), row.get("delivery_date_to_consignee"), 90),
#         ("<strong>Empty Container Returned to</strong>", row.get("empty_container_return_lcn"), row.get("empty_container_return_date"), 100),
#     ]

#     milestone_rows = []
#     for event, location, raw_date, rank in milestone_defs:
#         dt = pd.to_datetime(raw_date, errors="coerce")
#         if pd.isna(dt):
#             continue
#         milestone_rows.append(
#             {
#                 "event": event,
#                 "location": None if pd.isna(location) else location,
#                 "date": dt.strftime("%Y-%m-%d"),
#                 "_dt": dt,
#                 "_rank": rank,
#             }
#         )

#     if not milestone_rows:
#         return f"No milestones found for container {container_no}."

#     milestones_df = pd.DataFrame(milestone_rows)

#     # Sort chronologically for display
#     # milestones_df = milestones_df.sort_values("_dt", ascending=True)

#     # Pick "latest status" by (rank first, then date)
#     last_row = max(milestone_rows, key=lambda x: (x["_rank"], x["_dt"]))
#     latest_text = f"The Container <con>{container_no}</con> {last_row['event']} {last_row['location']} on {last_row['date']}"

#     # Convert milestone dataframe to string (no internal helper cols)
#     milestone_text = milestones_df[["event", "location", "date"]].to_string(index=False, header=False)

#     result = (
#         f"{header_text}"
#         f"{latest_text}\n\n"
#         f" <MILESTONE> {milestone_text}."
#     )
#     return result

def safe_date(v):
    """
    Safely convert a value to date in YYYY-MM-DD format.
    Returns None if value is invalid or empty.
    """

    if pd.isna(v) or not v:
        return None
    try:
        dt = pd.to_datetime(v, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.strftime("%Y-%m-%d")  # ✅ consistent ISO format
    except Exception:
        return str(v)


def get_top_values_for_column(query: str) -> str:
    """
    Get the top 5 most frequent values for a specified column.
    Input: Query should mention a column name or field.
    Output: Top 5 values with their frequency counts.
    """
    # Extract column name from query using fuzzy matching
    df = _df()
    words = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_]+\b", query)
             if w.lower() not in {"what", "are", "the", "top", "values", "for", "in", "of"}]

    if not words:
        return "Please specify a column name to get top values for."

    # Find best matching column
    best_matches = []
    for word in words:
        if len(word) > 2:
            match = process.extractOne(word, df.columns.tolist())
            if match and match[1] > 70:
                best_matches.append(match[0])

    if not best_matches:
        return "Could not find a matching column. Please specify a valid column name."

    column = best_matches[0]

    # Get top 5 values
    top_values = df[column].value_counts().head(5)

    if top_values.empty:
        return f"No data available for column '{column}'."

    result_lines = [f"Top 5 values for '{column}':"]
    for value, count in top_values.items():
        result_lines.append(f"  {value}: {count} occurrences")

    return "\n".join(result_lines)

def get_load_port_for_container(input_str: str) -> str:
    """
    Get the load port details for a specific container.
    Input: Provide a valid container number (partial or full).
    Output: Load port and related departure information.
    """
    container_no = extract_container_number(input_str)
    if not container_no:
        return "Please specify a valid container number."

    df = _df()
    df["container_number"] = df["container_number"].astype(str)

    # Exact match after normalizing
    clean = clean_container_number(container_no)
    rows = df[df["container_number"].str.replace(" ", "").str.upper() == clean]

    # Fallback to contains match
    if rows.empty:
        rows = df[df["container_number"].str.contains(container_no, case=False, na=False)]

    if rows.empty:
        return f"No data found for container {container_no}."

    row = rows.iloc[0]

    # Build load port information
    info_lines = [f"Load port information for container {container_no}:"]

    if pd.notnull(row.get('load_port')):
        info_lines.append(f"Load Port: {row['load_port']}")

    if pd.notnull(row.get('final_load_port')):
        info_lines.append(f"Final Load Port: {row['final_load_port']}")

    if pd.notnull(row.get('etd_lp')):
        etd = row['etd_lp']
        if hasattr(etd, 'strftime'):
            etd_str = etd.strftime('%Y-%m-%d')
        else:
            etd_str = str(etd)
        info_lines.append(f"ETD from Load Port: {etd_str}")

    if pd.notnull(row.get('atd_lp')):
        atd = row['atd_lp']
        if hasattr(atd, 'strftime'):
            atd_str = atd.strftime('%Y-%m-%d')
        else:
            atd_str = str(atd)
        info_lines.append(f"ATD from Load Port: {atd_str}")

    if len(info_lines) == 1:  # Only header line
        info_lines.append("No load port information available.")

    return "\n".join(info_lines)

def answer_with_column_mapping(query: str) -> str:
    """
    Interprets user queries using robust synonym mapping and answers using the correct column(s).
    Input: Natural language query about shipment data.
    Output: Answer based on mapped columns and data analysis.
    """
    from agents.prompts import map_synonym_to_column

    df = _df()

    # Try to map query terms to columns
    query_lower = query.lower()
    mapped_columns = []

    # Common query patterns and their column mappings
    column_mappings = {
        'consignee': 'consignee_code_multiple',
        'po number': 'po_number_multiple',
        'container number': 'container_number',
        'vessel': 'final_vessel_name',
        'carrier': 'final_carrier_name',
        'load port': 'load_port',
        'discharge port': 'discharge_port',
        'eta': 'eta_dp',
        'ata': 'ata_dp',
        'etd': 'etd_lp',
        'atd': 'atd_lp'
    }

    # Find relevant columns based on query
    for term, column in column_mappings.items():
        if term in query_lower and column in df.columns:
            mapped_columns.append(column)

    if not mapped_columns:
        return "Could not map your query to specific data columns. Please be more specific."

    # Extract specific values if mentioned (container numbers, PO numbers, etc.)
    container_no = extract_container_number(query)

    if container_no:
        # Query about specific container
        rows = df[df["container_number"].astype(str).str.contains(container_no, case=False, na=False)]
        if rows.empty:
            return f"No data found for container {container_no}."

        row = rows.iloc[0]
        result_lines = [f"Information for container {container_no}:"]

        for col in mapped_columns:
            if col in row.index and pd.notnull(row[col]):
                value = row[col]
                if pd.api.types.is_datetime64_dtype(df[col]) or isinstance(value, pd.Timestamp):
                    value = value.strftime("%Y-%m-%d")
                result_lines.append(f"{col.replace('_', ' ').title()}: {value}")

        return "\n".join(result_lines)
    else:
        # General query about the columns
        result_lines = []
        for col in mapped_columns[:3]:  # Limit to 3 columns
            if pd.api.types.is_datetime64_dtype(df[col]):
                non_null = df[col].dropna()
                if not non_null.empty:
                    result_lines.append(f"{col.replace('_', ' ').title()}:")
                    result_lines.append(f"  Date range: {non_null.min().date()} to {non_null.max().date()}")
                    result_lines.append(f"  Total records: {non_null.count()}")
            else:
                top_values = df[col].value_counts().head(3)
                if not top_values.empty:
                    result_lines.append(f"{col.replace('_', ' ').title()} (top values):")
                    for val, count in top_values.items():
                        result_lines.append(f"  {val}: {count}")

        return "\n".join(result_lines) if result_lines else "No data available for the specified fields."

# 2️⃣ Delayed Containers (X days)
def ensure_datetime(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Convert specified columns to datetime using explicit formats to avoid inference warnings."""
    # Common date/time formats seen in shipment data
    known_formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%m/%d/%Y %H:%M",
        "%d/%m/%Y %H:%M",
        "%d-%m-%Y",
        "%d-%m-%Y %H:%M:%S",
        "%d-%b-%Y",              # e.g., 22-May-2025
        "%d-%b-%Y %H:%M:%S",
        "%m/%d/%Y %I:%M:%S %p",
    ]

    for col in columns:
        if col not in df.columns:
            continue

        # Already datetime-like → skip
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue

        # Work on a copy; standardize blanks
        s = df[col].astype(str).str.strip().replace({"": None, "NaN": None, "nan": None, "NaT": None})

        # Start with all NaT and fill in progressively using explicit formats
        parsed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        for fmt in known_formats:
            mask = parsed.isna()
            if not mask.any():
                break
            try:
                parsed.loc[mask] = pd.to_datetime(s[mask], format=fmt, errors="coerce")
            except Exception:
                # Ignore bad format
                continue

        # Assign parsed results (no generic inference → no warning)
        df[col] = parsed

        # Optional: log how many failed to parse
        try:
            fail_count = int(parsed.isna().sum())
            if fail_count:
                logger.debug(f"ensure_datetime: column '{col}' has {fail_count} unparsed values")
        except Exception:
            pass

    return df


def get_delayed_containers(question: str = None, consignee_code: str = None, **kwargs) -> str:
    """
    Get containers that arrived delayed (ATA > ETA at discharge port).
    Supports:
    - Delay thresholds: "more than N days", "at least N days", "exactly N days", etc.
    - Time windows: "this week", "last 7 days", "in August", etc. (applied to ATA)
    - Port filtering: "delayed at USNYC", "late at Rotterdam"
    - Consignee filtering via consignee_code parameter

    Returns list[dict] with: container_number, eta_dp, ata_dp, delay_days, discharge_port, consignee_code_multiple
    """

    query = (question or "").strip()
    q_up = query.upper()

    try:
        logger.info(f"[get_delayed_containers] Query: {query!r}, consignee_code: {consignee_code}")
    except Exception:
        pass

    # 1) Parse time window using centralized helper
    start_date, end_date, period_desc = parse_time_period(query)
    apply_time_filter = False

    time_keywords = [
        r'\bthis\s+week\b', r'\bnext\s+week\b', r'\blast\s+week\b',
        r'\bthis\s+month\b', r'\bnext\s+month\b', r'\blast\s+month\b',
        r'\btoday\b', r'\btomorrow\b', r'\byesterday\b',
        r'\bnext\s+\d+\s+days?\b', r'\blast\s+\d+\s+days?\b',
        r'\bin\s+\d+\s+days?\b', r'\bwithin\s+\d+\s+days?\b',
        r'\bfrom\s+\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}\b',
        r'\bbetween\s+\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}\b'
    ]
    for pattern in time_keywords:
        if re.search(pattern, query, re.IGNORECASE):
            apply_time_filter = True
            break

    # Month parsing (applied to ATA window): handle both "in October" and "in October 2025".
    # NOTE: parse_time_period("October 2025") returns a single day (Oct 1) by default, so we
    # explicitly expand month+year to the full calendar month here.
    month_match = re.search(
        r"\b(?:in|during|on)\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:\s+(\d{4}))?\b",
        query,
        re.IGNORECASE,
    )
    requested_month = None
    if month_match:
        requested_month = (month_match.group(1) or "").lower()
        year_str = month_match.group(2)
        apply_time_filter = True

        today = pd.Timestamp.today().normalize()
        month_map = {
            'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
            'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6,
            'jul': 7, 'july': 7, 'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
            'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12
        }
        m = month_map.get(requested_month)
        if m:
            year = int(year_str) if year_str else today.year
            start_date = pd.Timestamp(year=year, month=m, day=1).normalize()
            if m == 12:
                end_date = pd.Timestamp(year=year + 1, month=1, day=1).normalize() - pd.Timedelta(days=1)
            else:
                end_date = pd.Timestamp(year=year, month=m + 1, day=1).normalize() - pd.Timedelta(days=1)

            # Nice display: "October 2025" (or "Oct 2025")
            month_disp = month_match.group(1)
            month_disp = month_disp.capitalize() if month_disp else requested_month.capitalize()
            period_desc = f"{month_disp} {year}"

    try:
        if apply_time_filter:
            logger.info(
                f"[get_delayed_containers] Time filter: {period_desc} "
                f"({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})"
            )
    except Exception:
        pass

    # 2) Load data
    df = _df()
    if df.empty:
        return "No container records available."

    # 3) Apply consignee filter if provided
    if consignee_code and "consignee_code_multiple" in df.columns:
        codes = [c.strip().upper() for c in str(consignee_code).split(",") if c.strip()]
        code_set = set(codes) | {c.lstrip("0") for c in codes}

        def row_has_code(cell):
            if pd.isna(cell):
                return False
            s = str(cell).upper()
            if any(re.search(rf"\({re.escape(c)}\)", s) for c in code_set):
                return True
            return any(re.search(rf"\b{re.escape(c)}\b", s) for c in code_set)

        df = df[df["consignee_code_multiple"].apply(row_has_code)].copy()
        if df.empty:
            return f"No containers found for consignee code(s) {', '.join(codes)}."

    # 4) Validate required columns
    if "ata_dp" not in df.columns or "eta_dp" not in df.columns:
        return "Required columns (ata_dp, eta_dp) not found in the dataset."

    # 5) Parse dates
    df = ensure_datetime(df, ["ata_dp", "eta_dp"])

    # 6) Filter to arrived containers only
    arrived_mask = df["ata_dp"].notna() & df["eta_dp"].notna()
    df_arrived = df[arrived_mask].copy()
    if df_arrived.empty:
        return "No arrived containers found."

    # 7) Apply time window filter on ATA (if detected)
    if apply_time_filter:
        ata_norm = df_arrived["ata_dp"].dt.normalize()
        time_mask = (ata_norm >= start_date) & (ata_norm <= end_date)
        df_arrived = df_arrived[time_mask].copy()
        if df_arrived.empty:
            return f"No containers arrived in {period_desc}."

        try:
            logger.info(f"[get_delayed_containers] After time filter: {len(df_arrived)} containers")
        except Exception:
            pass

    # 8) Calculate delay_days
    df_arrived["delay_days"] = (
        (df_arrived["ata_dp"] - df_arrived["eta_dp"]).dt.total_seconds() / 86400
    ).round().astype(int)

    # 9) Parse delay threshold from query
    delay_threshold = None
    delay_operator = ">"
    patterns = [
        (r'(?:less\s+than|under|below|<)\s*(\d+)\s+days?', '<'),
        (r'(?:more\s+than|over|above|greater\s+than|>)\s*(\d+)\s+days?', '>'),
        (r'(?:at\s+least|minimum|>=\s*|≥\s*)\s*(\d+)\s+days?', '>='),
        (r'(?:up\s+to|no\s+more\s+than|maximum|within|<=\s*|≤\s*)\s*(\d+)\s+days?', '<='),
        (r'(?:exactly|equal\s+to|=\s*)\s*(\d+)\s+days?', '=='),
        (r'(?:delayed\s+by|late\s+by)\s+(\d+)\s+days?(?:\s+late)?', '=='),
    ]
    for pattern, op in patterns:
        m = re.search(pattern, query, re.IGNORECASE)
        if m:
            delay_threshold = int(m.group(1))
            delay_operator = op
            break

    # 10) Apply delay filter
    if delay_threshold is not None:
        if delay_operator == '<':
            delay_mask = (df_arrived["delay_days"] > 0) & (df_arrived["delay_days"] < delay_threshold)
        elif delay_operator == '>':
            delay_mask = df_arrived["delay_days"] > delay_threshold
        elif delay_operator == '>=':
            delay_mask = df_arrived["delay_days"] >= delay_threshold
        elif delay_operator == '==':
            delay_mask = df_arrived["delay_days"] == delay_threshold
        elif delay_operator == '<=':
            delay_mask = (df_arrived["delay_days"] > 0) & (df_arrived["delay_days"] <= delay_threshold)
        else:
            delay_mask = df_arrived["delay_days"] > 0
    else:
        delay_mask = df_arrived["delay_days"] > 0

    result = df_arrived[delay_mask].copy()
    if result.empty:
        time_context = f" in {period_desc}" if apply_time_filter else ""
        if delay_threshold is not None:
            return f"No containers with delay {delay_operator} {delay_threshold} days{time_context}."
        else:
            return f"No delayed containers found{time_context}."

    # 11) Port filtering - **CRITICAL FIX**: Skip port filter if month was detected
    if "discharge_port" in result.columns and not requested_month:
        port_code = None
        port_name = None

        m_code = re.search(r"\(([A-Z0-9]{3,6})\)", q_up)
        if m_code:
            port_code = m_code.group(1).strip().upper()

        if not port_code:
            m_name = re.search(
                r"\b(?:AT|IN|FROM)\s+([A-Z][A-Z\s\.\-]{3,}?)(?:\s+BUT\s+|\s+AND\s+|,|\?|\.$|\s*$)",
                q_up
            )
            if m_name:
                port_name = m_name.group(1).strip()

        if port_code or port_name:
            def normalize_port(s):
                if pd.isna(s):
                    return ""
                t = str(s).upper()
                t = re.sub(r"\([^)]*\)", "", t)
                return re.sub(r"\s+", " ", t).strip()

            port_series = result["discharge_port"].astype(str)

            if port_code:
                port_mask = port_series.str.upper().str.contains(
                    rf"\({re.escape(port_code)}\)", na=False
                )
            else:
                port_norm = result["discharge_port"].apply(normalize_port)
                phrase_norm = re.sub(r"\s+", " ", port_name).strip()
                exact = (port_norm == phrase_norm)
                if exact.any():
                    port_mask = exact
                else:
                    words = [w for w in phrase_norm.split() if len(w) >= 3]
                    if words:
                        port_mask = pd.Series(True, index=result.index)
                        for w in words:
                            port_mask &= port_norm.str.contains(re.escape(w), na=False)
                    else:
                        port_mask = port_norm.str.contains(re.escape(phrase_norm), na=False)

            result = result[port_mask].copy()

            if result.empty:
                location = port_code or port_name
                time_context = f" in {period_desc}" if apply_time_filter else ""
                return f"No delayed containers at {location}{time_context}."

    # 12) Sort and format output
    if 'ata_dp' in result.columns:
        result = result.sort_values(['ata_dp', 'delay_days'], ascending=[False, False])
    else:
        result = result.sort_values("delay_days", ascending=False)

    out_cols = [
        "container_number",
        "eta_dp",
        "ata_dp",
        "delay_days",
        "discharge_port",
        "consignee_code_multiple",
    ]
    additional_cols = ["po_number_multiple", "final_carrier_name"]
    for col in additional_cols:
        if col in result.columns:
            out_cols.append(col)

    out_cols = [c for c in out_cols if c in result.columns]
    out = result[out_cols].head(200).copy()

    for dcol in ["eta_dp", "ata_dp"]:
        if dcol in out.columns and pd.api.types.is_datetime64_any_dtype(out[dcol]):
            out[dcol] = out[dcol].dt.strftime("%Y-%m-%d")

    return out.where(pd.notnull(out), None).to_dict(orient="records")


def get_delayed_bls(question: str = None, consignee_code: str = None, **kwargs) -> str:
    """
    Find delayed shipments by ocean BL (ocean_bl_no_multiple).
    Behaviour mirrors get_delayed_pos:
      - Supports numeric filters (exact, range, <, >, X+)
      - Supports consignee_code filtering (comma-separated) and consignee name detection
      - Supports location filtering (port code/name/fuzzy)
      - Accepts BL tokens in the query (alphanumeric 4–20 chars, comma-separated in dataset)
      - If no days mentioned, defaults to 'delayed by at least 7 days'
    Returns list[dict] with columns like:
      ocean_bl_no_multiple, container_number, eta_dp, revised_eta, ata_dp, delay_days, consignee_code_multiple, discharge_port
    NOTE: For arrived rows, delay is computed as:
      - ata_dp - revised_eta (if revised_eta column exists and value present)
      - otherwise ata_dp - eta_dp
    For not-yet-arrived rows delay is computed as today - eta_dp (unchanged).
    """
    try:
        from rapidfuzz import fuzz, process
    except Exception:
        fuzz = process = None

    query = (question or "")
    df = _df()

    # store requested consignee codes (uppercased) for later validation at BL-aggregation time
    requested_codes = []
    if consignee_code:
        requested_codes = [c.strip().upper() for c in str(consignee_code).split(",") if c.strip()]

    # find BL column robustly
    try:
        bl_col = _find_ocean_bl_col(df) or ("ocean_bl_no_multiple" if "ocean_bl_no_multiple" in df.columns else None)
    except Exception:
        bl_col = "ocean_bl_no_multiple" if "ocean_bl_no_multiple" in df.columns else None

    if not bl_col or bl_col not in df.columns:
        return "Ocean BL column (ocean_bl_no_multiple) not found in dataset."

    # date columns used to compute delay -- include revised_eta variants
    date_cols = ["eta_dp", "ata_dp", "predictive_eta_fd", "revised_eta", "revised_eta_fd", "eta_fd"]
    df = ensure_datetime(df, [c for c in date_cols if c in df.columns])

    # detect which revised column exists (prefer revised_eta)
    revised_col = None
    for cand in ("revised_eta", "revised_eta_fd"):
        if cand in df.columns:
            revised_col = cand
            break

    # Apply consignee filter if provided (row-level scoping)
    if requested_codes and "consignee_code_multiple" in df.columns:
        mask = pd.Series(False, index=df.index)
        for c in requested_codes:
            mask |= df["consignee_code_multiple"].astype(str).str.upper().str.contains(re.escape(c), na=False)
        df = df[mask].copy()
    if df.empty:
        return "No BL records found for provided consignee codes."

    # Detect consignee name in question and filter (best-effort)
    consignee_name_filter = None
    if "consignee_code_multiple" in df.columns:
        all_names = df["consignee_code_multiple"].dropna().astype(str).unique().tolist()
        q_up = query.upper()
        for name in all_names:
            clean_name = re.sub(r"\([^)]*\)", "", name).strip().upper()
            if clean_name and clean_name in q_up:
                consignee_name_filter = clean_name
                break
        if consignee_name_filter:
            df = df[df["consignee_code_multiple"].astype(str).str.upper().str.contains(consignee_name_filter)]
            if df.empty:
                return f"No BL records found for consignee '{consignee_name_filter}'."

    # Compute delay_days:
    #  - For rows with ata_dp: use revised_eta if present for that row else fallback to eta_dp
    #  - For rows without ata_dp but with eta: use today - eta (overdue)
    today = pd.Timestamp.now().normalize()

    # select primary ETA candidate column (eta_dp preferred)
    eta_col = None
    if "eta_dp" in df.columns:
        eta_col = "eta_dp"
    elif "eta_fd" in df.columns:
        eta_col = "eta_fd"

    ata_col = "ata_dp" if "ata_dp" in df.columns else ("predictive_eta_fd" if ("predictive_eta_fd" in df.columns and "eta_fd" in df.columns) else None)

    # default zero if no eta at all
    if eta_col is None:
        df["delay_days"] = pd.Series(0, index=df.index)
    else:
        df["delay_days"] = pd.NA
        # arrived rows: compute using revised_eta (row-level) when available else eta_col
        if ata_col and ata_col in df.columns:
            mask_arrived = df[ata_col].notna() & (df[eta_col].notna() | (revised_col is not None and df[revised_col].notna()))
            if mask_arrived.any():
                if revised_col:
                    baseline = df[revised_col].where(df[revised_col].notna(), df[eta_col])
                else:
                    baseline = df[eta_col]
                df.loc[mask_arrived, "delay_days"] = (df.loc[mask_arrived, ata_col] - baseline.loc[mask_arrived]).dt.days
        # not-yet-arrived rows: overdue relative to today
        mask_not_arrived = (~df[eta_col].isna()) & (~(df[ata_col].notna() if (ata_col and ata_col in df.columns) else False))
        df.loc[mask_not_arrived, "delay_days"] = (today - df.loc[mask_not_arrived, eta_col]).dt.days

        # finalize
        df["delay_days"] = df["delay_days"].fillna(0).astype(int)

    arrived = df.copy()

    # BL token detection (if user referenced a BL directly)
    q_up = query.upper()
    query_bl_tokens = set(re.findall(r'\b[A-Z0-9]{4,20}\b', q_up))
    matched_bl_norms = set()
    matched_originals = set()
    if query_bl_tokens:
        norm_to_originals = {}
        for raw in arrived[bl_col].dropna().astype(str).tolist():
            for part in re.split(r',\s*', raw):
                p = part.strip()
                if not p:
                    continue
                try:
                    norm = _normalize_bl_token(p)
                except Exception:
                    norm = re.sub(r'[^A-Z0-9]', '', p.upper())
                if not norm:
                    continue
                norm_to_originals.setdefault(norm, set()).add(p.upper())
        for tok in query_bl_tokens:
            tok_norm = re.sub(r'[^A-Z0-9]', '', tok.upper())
            if tok_norm in norm_to_originals:
                matched_bl_norms.add(tok_norm)
                matched_originals.update(norm_to_originals.get(tok_norm, set()))
            else:
                if process and len(tok) >= 4:
                    all_orig = list({o for s in norm_to_originals.values() for o in s})
                    if all_orig:
                        best = process.extractOne(tok, all_orig, scorer=fuzz.token_set_ratio, score_cutoff=85)
                        if best:
                            cand = re.sub(r'[^A-Z0-9]', '', best[0].upper())
                            if cand in norm_to_originals:
                                matched_bl_norms.add(cand)
                                matched_originals.update(norm_to_originals.get(cand, set()))
    if matched_bl_norms:
        def row_has_norm_bl(cell):
            if pd.isna(cell):
                return False
            parts = [p.strip() for p in re.split(r',\s*', str(cell)) if p.strip()]
            for p in parts:
                try:
                    n = _normalize_bl_token(p)
                except Exception:
                    n = re.sub(r'[^A-Z0-9]', '', p.upper())
                if n in matched_bl_norms:
                    return True
            return False
        arrived = arrived[arrived[bl_col].apply(row_has_norm_bl)].copy()
        if arrived.empty:
            return f"No delayed BLs matching {sorted(list(matched_originals or matched_bl_norms))} for your authorized consignees."

    # Location filter (code, name, or fuzzy)
    port_cols = [c for c in ["discharge_port", "final_destination", "place_of_delivery", "load_port"] if c in arrived.columns]

    def _extract_loc_code_and_name_for_bl(q: str):
        q_up = (q or "").upper()
        m = re.search(r"\(([A-Z0-9]{3,6})\)", q_up)
        if m:
            return m.group(1), None
        cand_codes = set(re.findall(r"\b[A-Z0-9]{3,6}\b", q_up))
        if port_cols and cand_codes:
            known_codes = set()
            for c in port_cols:
                vals = arrived[c].dropna().astype(str).str.upper()
                known_codes |= set(re.findall(r"\(([A-Z0-9]{3,6})\)", " ".join(vals.tolist())))
            for code in cand_codes:
                if code in known_codes:
                    return code, None
        m2_list = re.findall(r"(?:\b(?:ON|AT|IN|TO|FROM)\s+([A-Z][A-Z0-9\s\.,'\-]{2,}))", q_up)
        if m2_list:
            cand = max(m2_list, key=len).strip()
            cand = re.sub(r"(?:\d+\s*DAYS?|DELAY|LATE|BEHIND|ETA|BY).*", "", cand).strip()
            if cand:
                return None, cand
        if port_cols and process:
            all_ports = set()
            for c in port_cols:
                vals = arrived[c].dropna().astype(str)
                vals = vals.str.replace(r"\([^)]*\)", "", regex=True).str.strip().str.upper().tolist()
                all_ports.update(vals)
            if all_ports:
                best = process.extractOne(q_up, list(all_ports), scorer=fuzz.token_set_ratio, score_cutoff=85)
                if best:
                    return None, best[0]
        return None, None

    code, name = _extract_loc_code_and_name_for_bl(query)
    if code or name:
        loc_mask = pd.Series(False, index=arrived.index)
        if code:
            for c in port_cols:
                loc_mask |= arrived[c].astype(str).str.upper().str.contains(rf"\({re.escape(code)}\)", na=False)
        else:
            tokens = [t for t in re.split(r"\W+", (name or "")) if len(t) >= 3]
            for c in port_cols:
                col_vals = arrived[c].astype(str).str.upper()
                cond = pd.Series(True, index=arrived.index)
                for t in tokens:
                    cond &= col_vals.str.contains(re.escape(t), na=False)
                loc_mask |= cond
        arrived = arrived[loc_mask].copy()
        if arrived.empty:
            where = f"{code or name}"
            return f"No delayed BLs found at {where} for your authorized consignees."

    # Delay day filters (order: range, <, >, X+, >=, ==, default >=7)
    q = query.lower()
    range_match = re.search(r"(\d+)\s*[-–—]\s*(\d+)\s*days?", q)
    less_than = re.search(r"(?:less\s+than|under|below|<)\s*(\d+)\s*days?", q)
    more_than = re.search(r"(?:more\s+than|over|above|>\s*)(\d+)\s*days?", q)
    plus_sign = re.search(r"\b(\d+)\s*\+\s*days?\b", q)
    at_least = re.search(r"(?:at\s+least|>=|minimum)\s*(\d+)\s*days?", q)
    exact = re.search(r"(?:exactly|by|of|in)\s+(\d+)\s+days?", q)

    if range_match:
        d1, d2 = int(range_match.group(1)), int(range_match.group(2))
        low, high = min(d1, d2), max(d1, d2)
        delayed = arrived[(arrived["delay_days"] >= low) & (arrived["delay_days"] <= high)]
    elif less_than:
        d = int(less_than.group(1))
        delayed = arrived[(arrived["delay_days"] > 0) & (arrived["delay_days"] < d)]
    elif more_than:
        d = int(more_than.group(1))
        delayed = arrived[arrived["delay_days"] > d]  # strictly greater than
    elif plus_sign:
        d = int(plus_sign.group(1))
        delayed = arrived[arrived["delay_days"] >= d]
    elif at_least:
        d = int(at_least.group(1))
        delayed = arrived[arrived["delay_days"] >= d]
    elif exact:
        d = int(exact.group(1))
        delayed = arrived[arrived["delay_days"] == d]
    else:
        delayed = arrived[arrived["delay_days"] >= 7]

    if delayed.empty:
        where = f" at {code or name}" if (code or name) else ""
        return f"No delayed BLs found for your authorized consignees{where}."

    # Output formatting & aggregation per BL
    # Ensure aggregated BL groups include requested consignee(s) if supplied
    # include revised_col in out_cols if available
    out_cols = [bl_col, "container_number", "eta_dp", "ata_dp", "delay_days",
                "consignee_code_multiple", "discharge_port"]
    if revised_col and revised_col in delayed.columns:
        # put revised column right after eta_dp for readability
        idx = out_cols.index("eta_dp") + 1
        out_cols.insert(idx, revised_col)
    out_cols = [c for c in out_cols if c in delayed.columns]

    if bl_col in delayed.columns:
        agg = delayed[out_cols].copy()
        # build agg dict dynamically to include revised_col when present
        agg_dict = {
            "container_number": lambda s: ", ".join(sorted(set(s.dropna().astype(str)))),
            "delay_days": "max",
            "eta_dp": "first",
            "ata_dp": "first",
            "consignee_code_multiple": lambda s: ", ".join(sorted(set([str(x).strip() for x in s.dropna().astype(str)]))),
            "discharge_port": "first"
        }
        if revised_col and revised_col in agg.columns:
            agg_dict[revised_col] = "first"

        agg_group = agg.groupby(bl_col).agg(agg_dict).reset_index()

        # If user asked for specific consignee codes, keep only BL groups that contain any of them
        if requested_codes and "consignee_code_multiple" in agg_group.columns:
            def group_has_requested_codes(s):
                s_up = (s or "").upper()
                for rc in requested_codes:
                    if re.search(re.escape(rc), s_up):
                        return True
                return False
            agg_group = agg_group[agg_group["consignee_code_multiple"].apply(group_has_requested_codes)].copy()
            if agg_group.empty:
                return f"No delayed BLs found for consignee code(s) {', '.join(requested_codes)}."

        # format date columns including revised_col
        for dcol in ["eta_dp", revised_col, "ata_dp"]:
            if dcol and dcol in agg_group.columns and pd.api.types.is_datetime64_any_dtype(agg_group[dcol]):
                agg_group[dcol] = agg_group[dcol].dt.strftime("%Y-%m-%d")

        # SORT by delay_days descending
        if "delay_days" in agg_group.columns:
            agg_group = agg_group.sort_values("delay_days", ascending=False).reset_index(drop=True)

        return agg_group.where(pd.notnull(agg_group), None).to_dict(orient="records")

    # fallback: row-level results (also sorted)
    out = delayed[out_cols].sort_values("delay_days", ascending=False).head(200).copy()

    # defensive final filter: if requested_codes present ensure rows contain them
    if requested_codes and "consignee_code_multiple" in out.columns:
        def row_has_code(cell):
            if pd.isna(cell):
                return False
            s = str(cell).upper()
            for rc in requested_codes:
                if re.search(re.escape(rc), s):
                    return True
            return False
        out = out[out["consignee_code_multiple"].apply(row_has_code)].copy()
        if out.empty:
            return f"No delayed BLs found for consignee code(s) {', '.join(requested_codes)}."

    for dcol in ["eta_dp", revised_col, "ata_dp"]:
        if dcol and dcol in out.columns and pd.api.types.is_datetime64_any_dtype(out[dcol]):
            out[dcol] = out[dcol].dt.strftime("%Y-%m-%d")

    return out.where(pd.notnull(out), None).to_dict(orient="records")


def get_hot_containers(question: str = None, consignee_code: str = None, **kwargs) -> str:
    """
    Unified hot-container handler.

    Enhancements:
      Supports consignee_code filtering (comma-separated)
      Detects consignee name in question to further narrow results
      Handles "less than", "more than", "8+", "1–3", "missed ETA", etc.
      Location detection now supports both codes (USLAX) and names ("Los Angeles")
    """

    query = (question or "")
    df = _df()  # already consignee-filtered if thread context applies

    # Apply consignee code filter
    if consignee_code and "consignee_code_multiple" in df.columns:
        codes = [c.strip() for c in str(consignee_code).split(",") if c.strip()]
        mask = pd.Series(False, index=df.index)
        for c in codes:
            mask |= df["consignee_code_multiple"].astype(str).str.contains(c)
        df = df[mask].copy()

    if df.empty:
        return "No container records found for provided consignee codes."

    # Identify hot-flag column
    hot_flag_cols = [c for c in df.columns if 'hot_container_flag' in c.lower()] or \
                    [c for c in df.columns if 'hot_container' in c.lower()]
    if not hot_flag_cols:
        return "Hot container flag column not found in the data."
    hot_flag_col = hot_flag_cols[0]

    def _is_hot(v) -> bool:
        if pd.isna(v):
            return False
        s = str(v).strip().upper()
        return s in {"Y", "YES", "TRUE", "1", "HOT"} or v is True or v == 1

    hot_df = df[df[hot_flag_col].apply(_is_hot)].copy()
    if hot_df.empty:
        return "No hot containers found for your authorized consignees."

    # Detect consignee name in question
    consignee_name_filter = None
    if "consignee_code_multiple" in hot_df.columns:
        all_names = hot_df["consignee_code_multiple"].dropna().astype(str).unique().tolist()
        q_up = query.upper()
        for name in all_names:
            clean_name = re.sub(r"\([^)]*\)", "", name).strip().upper()
            if clean_name and clean_name in q_up:
                consignee_name_filter = clean_name
                break
        if consignee_name_filter:
            hot_df = hot_df[hot_df["consignee_code_multiple"].astype(str).str.upper().str.contains(consignee_name_filter)]
            if hot_df.empty:
                return f"No hot containers found for consignee '{consignee_name_filter}'."

    # Location filters (enhanced)
    #port_cols = [c for c in ["discharge_port", "vehicle_arrival_lcn", "final_destination","place_of_delivery", "load_port"]
    port_cols = [c for c in ["discharge_port"]
                 if c in hot_df.columns]

    def _extract_loc_code_and_name(q: str):
        q_up = (q or "").upper()

        # (1) Code like (USLAX)
        m = re.search(r"\(([A-Z0-9]{3,6})\)", q_up)
        if m:
            return m.group(1), None

        # (2) Bare port code e.g., "USLAX" (check against known dataset codes)
        cand_codes = set(re.findall(r"\b[A-Z0-9]{3,6}\b", q_up))
        if port_cols and cand_codes:
            known_codes = set()
            for c in port_cols:
                vals = hot_df[c].dropna().astype(str).str.upper()
                known_codes |= set(re.findall(r"\(([A-Z0-9]{3,6})\)", " ".join(vals.tolist())))
            for code in cand_codes:
                if code in known_codes:
                    return code, None

        # (3) Named port/city with prepositions (on/at/in/to/from)
        m2_list = re.findall(r"(?:\b(?:ON|AT|IN|TO|FROM)\s+([A-Z][A-Z0-9\s\.,'\-]{2,}))", q_up)
        if m2_list:
            cand = max(m2_list, key=len).strip()
            # Stop at common separators like "FOR" (e.g., "at Los Angeles for consignee 0000866")
            cand = re.split(r"\bFOR\b", cand, maxsplit=1)[0].strip()
            # Clean noise like "BY", "DELAYED", "DAYS", etc.
            cand = re.sub(r"(?:(?:\d+\s*DAYS?)|ARRIV(?:ING|AL)?|LATE|DELAYED|OVERDUE|BEHIND|BY|ONWARD).*", "", cand).strip()
            if cand:
                return None, cand

        # (4) Fallback: match known port names in query (e.g., "Los Angeles")
        if port_cols:
            for c in port_cols:
                vals = hot_df[c].dropna().astype(str).str.upper().unique().tolist()
                for val in vals:
                    val_clean = re.sub(r"\([^)]*\)", "", val).strip()
                    if len(val_clean) > 2 and val_clean in q_up:
                        return None, val_clean

        return None, None

    code, name = _extract_loc_code_and_name(query)

    def _looks_like_time_phrase(text: str) -> bool:
        if not text:
            return False
        t = str(text).strip().lower()
        # Common month names/abbrevs, relative periods, or a year token.
        if re.search(
            r"\b(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|"
            r"sep|sept|september|oct|october|nov|november|dec|december)\b",
            t,
        ):
            return True
        if re.search(r"\b(this|next|last)\s+(week|month)\b", t):
            return True
        if re.search(r"\b(today|tomorrow|yesterday)\b", t):
            return True
        if re.search(r"\b\d{4}\b", t):
            return True
        return False

    # Avoid mis-classifying time phrases like "in November 2025" as locations.
    if name and _looks_like_time_phrase(name):
        name = None

    if code or name:
        loc_mask = pd.Series(False, index=hot_df.index)
        if code:
            for c in port_cols:
                loc_mask |= hot_df[c].astype(str).str.upper().str.contains(rf"\({re.escape(code)}\)", na=False)
        else:
            tokens = [t for t in re.split(r"\W+", (name or "")) if len(t) >= 3]
            for c in port_cols:
                col_vals = hot_df[c].astype(str).str.upper()
                cond = pd.Series(True, index=hot_df.index)
                for t in tokens:
                    cond &= col_vals.str.contains(re.escape(t), na=False)
                loc_mask |= cond
        hot_df = hot_df[loc_mask].copy()
        if hot_df.empty:
            where = f"{code or name}"
            return f"No hot containers found at {where} for your authorized consignees."

    ql = (query or "").lower()

    # Optional carrier/vessel filter (minimal + query-driven)
    def _extract_carrier_or_vessel(q: str) -> str | None:
        q_norm = (q or "").strip()
        if not q_norm:
            return None
        # Primary patterns: "with Maersk", "carrier Maersk", "line Maersk"
        m = re.search(
            r"\b(?:with|carrier|line)\s+([A-Za-z0-9&\-\.'\s]{2,60}?)(?=\s+\b(?:in|on|for|to|from|at|between|during|within|next|last|this)\b|\s*$)",
            q_norm,
            re.IGNORECASE,
        )
        if m:
            cand = m.group(1).strip(" \t\r\n,.;:-")
            return cand or None

        # Secondary pattern: "by <carrier>" (avoid delay phrases like "by more than 5 days")
        m2 = re.search(
            r"\bby\s+([A-Za-z][A-Za-z&\-\.'\s]{1,60}?)(?=\s+\b(?:in|on|for|to|from|at|between|during|within|next|last|this)\b|\s*$)",
            q_norm,
            re.IGNORECASE,
        )
        if not m2:
            return None
        cand2 = m2.group(1).strip(" \t\r\n,.;:-")
        # Reject common non-carrier phrases
        if re.search(r"\b(more|less|under|over|than|days?|weeks?|months?|eta|delayed|late|overdue|behind)\b", cand2, re.IGNORECASE):
            return None
        return cand2 or None

    carrier_or_vessel = _extract_carrier_or_vessel(query)
    if carrier_or_vessel:
        carrier_cols = [c for c in ["final_carrier_name", "final_vessel_name"] if c in hot_df.columns]
        if carrier_cols:
            carrier_mask = pd.Series(False, index=hot_df.index)
            for c in carrier_cols:
                carrier_mask |= hot_df[c].astype(str).str.contains(carrier_or_vessel, case=False, na=False)
            hot_df = hot_df[carrier_mask].copy()
            if hot_df.empty:
                return f"No hot containers found for carrier/vessel '{carrier_or_vessel}' for your authorized consignees."

    # Optional time filter (minimal + query-driven)
    def _extract_month_year_range(q: str) -> tuple[pd.Timestamp, pd.Timestamp, str] | None:
        ql_local = (q or "").lower()
        month_map = {
            "jan": 1,
            "january": 1,
            "feb": 2,
            "february": 2,
            "mar": 3,
            "march": 3,
            "apr": 4,
            "april": 4,
            "may": 5,
            "jun": 6,
            "june": 6,
            "jul": 7,
            "july": 7,
            "aug": 8,
            "august": 8,
            "sep": 9,
            "sept": 9,
            "september": 9,
            "oct": 10,
            "october": 10,
            "nov": 11,
            "november": 11,
            "dec": 12,
            "december": 12,
        }

        # Match "November 2025" or "Nov 2025" (month + year, no day).
        m = re.search(
            r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
            r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{4})\b",
            ql_local,
        )
        if not m:
            return None

        # If there's an explicit day present (e.g., "Nov 10 2025"), let parse_time_period handle it.
        if re.search(r"\b\d{1,2}\s+" + re.escape(m.group(1)) + r"\s+\d{4}\b", ql_local):
            return None
        if re.search(r"\b" + re.escape(m.group(1)) + r"\s+\d{1,2}\s+\d{4}\b", ql_local):
            return None

        month_key = m.group(1)
        year = int(m.group(2))
        # Normalize key to the shortest month prefix when possible.
        month_key = month_key[:3] if month_key[:3] in month_map else month_key
        month = month_map.get(month_key, month_map.get(month_key.lower()))
        if not month:
            return None

        start = pd.Timestamp(year=year, month=month, day=1).normalize()
        end = (start + pd.offsets.MonthEnd(0)).normalize()
        return start, end, f"{start.strftime('%B %Y')} ({start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')})"

    def _query_has_explicit_time(q: str) -> bool:
        ql_local = (q or "").lower()
        return bool(
            re.search(
                r"\b(today|tomorrow|yesterday)\b|\b(this|next|last)\s+(week|month)\b|\bbetween\b|\bfrom\b|"
                r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}\b|\b\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2}\b|"
                r"\b(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|"
                r"oct|october|nov|november|dec|december)\b\s+\d{4}\b|\bnext\s+\d{1,3}\s+days?\b|\blast\s+\d{1,3}\s+days?\b",
                ql_local,
            )
        )

    if _query_has_explicit_time(query):
        month_range = _extract_month_year_range(query)
        if month_range:
            start_date, end_date, period_desc = month_range
        else:
            start_date, end_date, period_desc = parse_time_period(query)

        eta_cols = [c for c in ["revised_eta", "eta_dp", "eta_fd", "predictive_eta"] if c in hot_df.columns]
        if eta_cols:
            hot_df = ensure_datetime(hot_df, eta_cols)
            eta_for_filter = hot_df[eta_cols].bfill(axis=1).iloc[:, 0]
            date_mask = eta_for_filter.notna() & (eta_for_filter.dt.normalize() >= start_date) & (eta_for_filter.dt.normalize() <= end_date)
            hot_df = hot_df[date_mask].copy()
            if hot_df.empty:
                return f"No hot containers found for {period_desc} for your authorized consignees."

    # A) Delayed / missed ETA hot containers
    if any(w in ql for w in ("delay", "late", "overdue", "behind", "missed", "eta", "deadline")):
        hot_df = ensure_datetime(hot_df, ["eta_dp", "ata_dp"])
        arrived = hot_df[hot_df["ata_dp"].notna()].copy()
        if arrived.empty:
            where = f" at {code or name}" if (code or name) else ""
            return f"No hot containers have arrived{where} for your authorized consignees."

        arrived["delay_days"] = (arrived["ata_dp"] - arrived["eta_dp"]).dt.days.fillna(0).astype(int)

        range_match = re.search(r"(\d{1,4})\s*[-–—]\s*(\d{1,4})\s*days?\b", ql, re.IGNORECASE)
        less_than = re.search(r"\b(?:less\s+than|under|below|<)\s*(\d{1,4})\s*days?\b", ql, re.IGNORECASE)
        more_than = re.search(r"\b(?:more\s+than|over|>)\s*(\d{1,4})\s*days?\b", ql, re.IGNORECASE)
        plus_sign = re.search(r"\b(\d{1,4})\s*\+\s*days?\b", ql, re.IGNORECASE)
        exact = re.search(r"\b(?:delayed|late|overdue|behind)\s+by\s+(\d{1,4})\s+days?\b", ql, re.IGNORECASE)

        if range_match:
            d1, d2 = int(range_match.group(1)), int(range_match.group(2))
            low, high = min(d1, d2), max(d1, d2)
            delayed = arrived[(arrived["delay_days"] >= low) & (arrived["delay_days"] <= high)]
        elif less_than:
            d = int(less_than.group(1))
            delayed = arrived[(arrived["delay_days"] > 0) & (arrived["delay_days"] < d)]
        elif more_than or plus_sign:
            d = int((more_than or plus_sign).group(1))
            delayed = arrived[arrived["delay_days"] > d]
        elif exact:
            d = int(exact.group(1))
            delayed = arrived[arrived["delay_days"] == d]
        else:
            delayed = arrived[arrived["delay_days"] > 0]

        delayed = delayed[delayed[hot_flag_col].apply(_is_hot)]
        delayed = delayed[delayed["delay_days"] > 0]

        if delayed.empty:
            where = f" at {code or name}" if (code or name) else ""
            return f"No hot containers are delayed for your authorized consignees{where}."

        cols = ["container_number", "eta_dp", "ata_dp", "delay_days", "discharge_port", "consignee_code_multiple"]
        #if "vehicle_arrival_lcn" in delayed.columns:
        #    cols.append("vehicle_arrival_lcn")
        cols = [c for c in cols if c in delayed.columns]

        out = delayed[cols].sort_values("delay_days", ascending=False).head(200).copy()
        for dcol in ["eta_dp", "ata_dp"]:
            if dcol in out.columns and pd.api.types.is_datetime64_any_dtype(out[dcol]):
                out[dcol] = out[dcol].dt.strftime("%Y-%m-%d")

        return out.where(pd.notnull(out), None).to_dict(orient="records")

    # C) Fallback - simple hot list
    display_cols = ['container_number', 'consignee_code_multiple']
    display_cols += [c for c in ['discharge_port', 'eta_dp', 'revised_eta'] if c in hot_df.columns]
    display_cols = [c for c in display_cols if c in hot_df.columns]

    if 'eta_dp' in hot_df.columns:
        hot_df = safe_sort_dataframe(hot_df, 'eta_dp', ascending=True)
    else:
        hot_df = safe_sort_dataframe(hot_df, 'container_number', ascending=True)

    result_data = hot_df[display_cols].head(200).copy()
    for col in result_data.columns:
        if pd.api.types.is_datetime64_dtype(result_data[col]):
            result_data[col] = result_data[col].dt.strftime('%Y-%m-%d')

    if len(result_data) == 0:
        return "No hot containers found for your authorized consignees."
    return result_data.where(pd.notnull(result_data), None).to_dict(orient="records")


def get_upcoming_arrivals(query: str) -> str:
    """
    List containers scheduled to arrive OR that have already arrived on specific dates.
    - Parses many natural forms for dates: "next N days", "today", "tomorrow", specific dates, date ranges
    - **NEW**: Centralized date parsing using parse_time_period()
    - **NEW**: Improved port/location filtering with multi-word port names support
    - Uses per-row ETA priority: revised_eta (if present) else eta_dp for FUTURE dates
    - For PAST dates, it uses ata_dp first, and if that is null, it falls back to derived_ata_dp
    - Respects consignee filtering via _df()
    - Strictly filters by discharge_port when a port code/city is provided (e.g., USLAX, Los Angeles)
    Returns list of dict records (up to 300 rows) or a short message if none found.
    """

    query = (query or "").strip()
    query_lower = query.lower()

    # Parse time period using centralized helper
    start_date, end_date, period_desc = parse_time_period(query)

    # Determine if it's a past query based on end_date
    is_past_query = (end_date < pd.Timestamp.today().normalize())

    try:
        logger.info(f"[get_upcoming_arrivals] Period: {period_desc}, "
                   f"Dates: {format_date_for_display(start_date)} to "
                   f"{format_date_for_display(end_date)}, is_past={is_past_query}")
    except Exception:
        pass

    # Load df and apply filters
    df = _df()

    # Location/Port filtering (IMPROVED - handles multi-word ports)
    port_cols = [c for c in ["discharge_port", "vehicle_arrival_lcn", "final_destination", "place_of_delivery"]
                 if c in df.columns]

    if port_cols:
        location_mask = pd.Series(False, index=df.index, dtype=bool)
        location_found = False
        location_name = None

        # Helper function to normalize port names
        def normalize_port_name(port_str):
            if pd.isna(port_str):
                return ""
            s = str(port_str).upper()
            # Remove content in parentheses (port codes)
            s = re.sub(r'\([^)]*\)', '', s)
            # Normalize whitespace
            s = re.sub(r'\s+', ' ', s).strip()
            return s

        # Pattern 1: Explicit port codes in parentheses like (USLAX) or (SGSIN)
        paren_match = re.search(r'\(([A-Z0-9]{3,6})\)', query.upper())
        if paren_match:
            tok = paren_match.group(1)
            tok_mask = pd.Series(False, index=df.index, dtype=bool)
            for col in port_cols:
                tok_mask |= df[col].astype(str).str.upper().str.contains(rf'\({re.escape(tok)}\)', na=False)
            if tok_mask.any():
                location_mask = tok_mask
                location_found = True
                location_name = tok
                try:
                    logger.info(f"[get_upcoming_arrivals] Found location in parentheses: {tok}")
                except:
                    pass

        # Pattern 2: City names with prepositions (explicit patterns)
        if not location_found:
            city_patterns = [
                # Pattern with prepositions: "at/in/to CITY_NAME"
                r'\b(?:AT|IN|TO)\s+([A-Z][A-Za-z\s\.\-\']{3,}?)(?:\s+IN\s+|\s+NEXT\s+|\s+WITHIN\s+|,|\s*$)',
                # Known major ports/cities
                r'\b(LOS\s+ANGELES|LONG\s+BEACH|NEW\s+YORK|SINGAPORE|ROTTERDAM|HONG\s+KONG|SHANGHAI|BUSAN|TOKYO|OAKLAND|SAVANNAH|HOUSTON|MIAMI|SEATTLE|CHICAGO|PORT\s+OF\s+[A-Z\s]+)\b'
            ]

            for pattern in city_patterns:
                city_match = re.search(pattern, query, re.IGNORECASE)
                if city_match:
                    city = city_match.group(1).strip()
                    # Clean up the matched city name by removing trailing timeframe words
                    city = re.sub(r'\s+(IN\s+)?NEXT.*$', '', city, flags=re.IGNORECASE).strip()
                    city = re.sub(r'\s+(IN\s+)?THE\s+LAST.*$', '', city, flags=re.IGNORECASE).strip()
                    city = re.sub(r'\s+LAST\s+.*$', '', city, flags=re.IGNORECASE).strip()

                    if city and len(city) > 2:
                        location_name = city.upper()

                        # Normalize and match
                        city_norm = normalize_port_name(city)
                        city_words = city_norm.split()

                        city_mask = pd.Series(False, index=df.index)
                        for col in port_cols:
                            # Create normalized column for matching
                            col_norm = df[col].apply(normalize_port_name)

                            # Try exact match first
                            exact_match = col_norm == city_norm

                            # If no exact match, try word-by-word matching
                            if not exact_match.any() and len(city_words) > 0:
                                word_match = pd.Series(True, index=df.index)
                                for word in city_words:
                                    if len(word) >= 3:  # Only check meaningful words
                                        word_match &= col_norm.str.contains(word, na=False, regex=False)
                                city_mask |= word_match
                            else:
                                city_mask |= exact_match

                        if city_mask.any():
                            location_mask = city_mask
                            location_found = True
                            try:
                                logger.info(f"[get_upcoming_arrivals] Found city location: {city} (normalized: {city_norm})")
                            except:
                                pass
                            break

        # Pattern 3: Bare port codes (only if no explicit location found) - validate against known codes
        if not location_found and port_cols:
            # Extract known port codes from the dataset FIRST
            known_codes = set()
            try:
                for col in port_cols:
                    vals = df[col].dropna().astype(str).str.upper()
                    known_codes |= set(re.findall(r'\(([A-Z0-9]{3,6})\)', ' '.join(vals.tolist())))
            except Exception:
                pass

            # Only proceed if we have known codes
            if known_codes:
                # Extract candidate tokens from query
                candidate_tokens = re.findall(r'\b[A-Z0-9]{3,6}\b', query.upper())

                # Skip ALL timeframe and common words
                skip_tokens = {
                    "NEXT", "DAYS", "DAY", "IN", "AT", "ON", "THE", "AND", "TO", "FROM",
                    "ARRIVE", "ARRIVING", "ARRIVED", "CONTAINERS", "CONTAINER", "PLEASE",
                    "CAN", "YOU", "LET", "ME", "KNOW", "WITHIN", "UPCOMING", "US", "TELL",
                    "TODAY", "TOMORROW", "WEEK", "MONTH", "THIS", "LAST", "AFTER", "BEFORE",
                    "SHOW", "LIST", "WHICH", "WHAT", "WHEN", "WHERE"
                }

                # Filter to only tokens that:
                # 1. Are not in skip list
                # 2. Are not purely numeric
                # 3. Actually exist in our known_codes set
                candidate_tokens = [
                    t for t in candidate_tokens
                    if t not in skip_tokens
                    and not t.isdigit()
                    and t in known_codes
                ]

                # Try each valid candidate
                for tok in candidate_tokens:
                    tok_mask = pd.Series(False, index=df.index)
                    for col in port_cols:
                        tok_mask |= df[col].astype(str).str.upper().str.contains(rf'\({re.escape(tok)}\)', na=False)

                    if tok_mask.any():
                        location_mask = tok_mask
                        location_found = True
                        location_name = tok
                        try:
                            logger.info(f"[get_upcoming_arrivals] Found bare port code: {tok} (validated against known codes)")
                        except:
                            pass
                        break

        # Apply location filter if found
        if location_found:
            df = df[location_mask].copy()
            if df.empty:
                verb = "arrived" if is_past_query else "scheduled to arrive"
                return f"No containers {verb} at {location_name} between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}."

    # ETA/ATA selection and date filtering
    if is_past_query:
        # --- PAST ARRIVALS LOGIC ---
        date_cols = [c for c in ['ata_dp', 'derived_ata_dp'] if c in df.columns]
        if not date_cols:
            return "No actual arrival date columns (ata_dp, derived_ata_dp) found."

        df = ensure_datetime(df, date_cols)

        # Create a single arrival date column, prioritizing ata_dp
        df['arrival_date_for_filter'] = pd.NaT
        if 'ata_dp' in df.columns:
            df['arrival_date_for_filter'] = df['ata_dp']
        if 'derived_ata_dp' in df.columns:
            df['arrival_date_for_filter'] = df['arrival_date_for_filter'].fillna(df['derived_ata_dp'])

        # Filter based on the combined arrival date column
        mask = (df['arrival_date_for_filter'].dt.normalize() >= start_date) & \
               (df['arrival_date_for_filter'].dt.normalize() <= end_date)

        result_df = df[mask].copy()
        sort_col = 'arrival_date_for_filter'
        output_cols = ['container_number', 'discharge_port', 'ata_dp']

    else:
        # --- FUTURE ARRIVALS LOGIC ---
        date_cols = [c for c in ['revised_eta', 'eta_dp', 'ata_dp'] if c in df.columns]
        if not any(c in date_cols for c in ['revised_eta', 'eta_dp']):
             return "No estimated arrival date columns (revised_eta, eta_dp) found."

        df = ensure_datetime(df, date_cols)

        # Create a single ETA column, prioritizing revised_eta
        df['eta_for_filter'] = pd.NaT
        if 'revised_eta' in df.columns:
            df['eta_for_filter'] = df['revised_eta']
        if 'eta_dp' in df.columns:
            df['eta_for_filter'] = df['eta_for_filter'].fillna(df['eta_dp'])

        # Filter for future dates and exclude already arrived containers
        mask = (df['eta_for_filter'].dt.normalize() >= start_date) & \
               (df['eta_for_filter'].dt.normalize() <= end_date)
        if 'ata_dp' in df.columns:
            mask &= df['ata_dp'].isna()

        result_df = df[mask].copy()
        sort_col = 'eta_for_filter'
        output_cols = ['container_number', 'discharge_port', 'revised_eta', 'eta_dp']

    # Format and return results
    if result_df.empty:
        verb = "arrived" if is_past_query else "scheduled to arrive"
        loc_str = f" at {location_name}" if location_found else ""
        if start_date == end_date:
            return f"No containers {verb}{loc_str} on {start_date.strftime('%Y-%m-%d')}."
        else:
            return f"No containers {verb}{loc_str} between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}."

    # Add common columns and sort
    if "consignee_code_multiple" in result_df.columns:
        output_cols.append("consignee_code_multiple")
    if "po_number_multiple" in result_df.columns:
        output_cols.append("po_number_multiple")

    final_cols = [c for c in output_cols if c in result_df.columns]
    out_df = result_df.sort_values(by=sort_col, ascending=True).head(300)[final_cols]

    # Format all date columns for clean output
    for col in out_df.select_dtypes(include=['datetime64[ns]']).columns:
        out_df[col] = out_df[col].dt.strftime('%Y-%m-%d')

    return out_df.where(pd.notnull(out_df), None).to_dict(orient='records')


def get_container_etd(query: str) -> str:
    """
    Return  ETD_LP(Estimated time of departure from Load Port) details for specific containers.
    Input: Query mentioning one or more container numbers (comma-separated or space-separated).
    Output: ETD_LP and port details for the containers.
    """
    # Extract all container numbers using regex pattern
    container_pattern = re.findall(r'([A-Z]{4}\d{7})', query)

    if not container_pattern:
        return "Please mention one or more container numbers."

    df = _df()

    # Add "MM/dd/yyyy hh:mm:ss tt" format to ensure_datetime function
    # or directly parse dates here
    #date_cols = ["eta_dp", "ata_dp"]
    date_cols = ["etd_lp"]
    for col in date_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass

    # Create results table with all requested containers
    results = []
    for cont in container_pattern:
        # First try exact match after normalizing
        norm_cont = cont.upper().strip()
        mask = df["container_number"].astype(str).str.upper().str.strip() == norm_cont
        row = df[mask]

        # If no exact match, try contains
        if row.empty:
            row = df[df["container_number"].astype(str).str.contains(cont, case=False, na=False)]

        if not row.empty:
            row = row.iloc[0]
            #cols = ["container_number", "discharge_port", "eta_dp", "ata_dp"]
            cols = ["container_number", "discharge_port", "etd_lp"]
            cols = [c for c in cols if c in row.index]
            single_result = row[cols].to_frame().T
            results.append(single_result)
        else:
            # Create a row with "Not Available" for missing containers
            missing_row = pd.DataFrame({
                "container_number": [cont],
                "discharge_port": ["Not Available"],
                "etd_lp": ["Not Available"],
                "etd_flp": ["Not Available"]
            })
            results.append(missing_row[["container_number", "discharge_port", "etd_lp"]])

    # Combine all results
    combined_results = pd.concat(results, ignore_index=True)

    # Format date columns (only for actual datetime values)
    for date_col in "etd_lp":
        if date_col in combined_results.columns:
            combined_results[date_col] = combined_results[date_col].apply(
                lambda x: x.strftime("%Y-%m-%d") if isinstance(x, pd.Timestamp) else x
            )

    return combined_results.to_string(index=False)


def get_arrivals_by_port(query: str) -> str:
    """
    Find containers arriving at a specific port or country within the next N days.
    Behaviour:
    - Parse timeframe using centralized parse_time_period() helper.
    - For each row, prefer revised_eta; if null use eta_dp (eta_for_filter).
    - Exclude rows that already have ata_dp (already arrived).
    - Return up to 50 matching rows (dict records) with formatted dates.
    """

    df = _df()

    # ---------- 1) Parse timeframe using centralized helper ----------
    start_date, end_date, period_desc = parse_time_period(query)

    try:
        logger.info(f"[get_arrivals_by_port] Period: {period_desc}, "
                   f"Dates: {format_date_for_display(start_date)} to "
                   f"{format_date_for_display(end_date)}")
    except:
        pass

    # ---------- 2) Extract port name or code ----------
    port_name_query = None
    port_code_query = None
    m_paren = re.search(r'([A-Za-z0-9\-\s\.\']+?)\s*\(([A-Z0-9]{2,6})\)', query)
    if m_paren:
        port_name_query = m_paren.group(1).strip()
        port_code_query = m_paren.group(2).strip().upper()
    else:
        m = re.search(
            r'(?:arriv(?:ing)?\s+(?:in|at|to)|in\s+|at\s+|port\s+)\s*([A-Za-z0-9\-\s\(\)\.]{2,60}?)\s*(?:,|for|in\s+next|within|next|\b\d+\s+days?\b|$)',
            query, re.IGNORECASE
        )
        if m:
            port_name_query = m.group(1).strip()
        else:
            caps = re.findall(r'\b([A-Z]{3,6})\b', query)
            if caps:
                port_code_query = caps[-1].strip().upper()
            else:
                tokens = re.findall(r'[A-Za-z0-9\-\.\']{3,}', query)
                port_name_query = tokens[-1] if tokens else ""

    if port_name_query:
        port_name_query = port_name_query.upper()

    # ---------- 3) Which port columns to check ----------
    preferred_cols = ['discharge_port', 'final_load_port']
    existing_port_cols = [c for c in preferred_cols if c in df.columns]
    if not existing_port_cols:
        return "No port-related columns found in the data."

    # ---------- 4) Build match mask ----------
    mask = pd.Series(False, index=df.index)
    if port_code_query:
        for col in existing_port_cols:
            mask |= df[col].astype(str).str.upper().str.contains(re.escape(port_code_query), na=False)
    else:
        port_choices = set()
        for col in existing_port_cols:
            port_choices.update(df[col].dropna().astype(str).str.upper().unique())
        if port_name_query in port_choices:
            for col in existing_port_cols:
                mask |= (df[col].astype(str).str.upper() == port_name_query)
        else:
            close = get_close_matches(port_name_query, list(port_choices), n=6, cutoff=0.6) if port_choices else []
            if close:
                for candidate in close:
                    for col in existing_port_cols:
                        mask |= df[col].astype(str).str.upper().str.contains(re.escape(candidate), na=False)
            else:
                words = [w for w in re.split(r'\W+', port_name_query or "") if len(w) >= 3]
                for w in words:
                    for col in existing_port_cols:
                        mask |= df[col].astype(str).str.upper().str.contains(re.escape(w), na=False)

    filtered = df[mask].copy()

    if filtered.empty:
        descriptor = port_code_query or port_name_query or "<unspecified>"
        return f"No containers found matching '{descriptor}' in the chosen port columns."

    # apply transport mode filter if present
    modes = extract_transport_modes(query)
    if modes and 'transport_mode' in filtered.columns:
        filtered = filtered[filtered['transport_mode'].astype(str).str.lower().apply(lambda s: any(m in s for m in modes))]

    # ---------- 5) Dates: per-row ETA selection ----------
    date_priority = [c for c in ['revised_eta', 'eta_dp'] if c in filtered.columns]
    if not date_priority:
        return "No ETA/arrival date columns found (expected 'revised_eta' or 'eta_dp')."

    parse_cols = date_priority.copy()
    if 'ata_dp' in filtered.columns:
        parse_cols.append('ata_dp')
    filtered = ensure_datetime(filtered, parse_cols)

    # Create per-row preferred ETA (revised_eta > eta_dp)
    if 'revised_eta' in filtered.columns and 'eta_dp' in filtered.columns:
        filtered['eta_for_filter'] = filtered['revised_eta'].where(filtered['revised_eta'].notna(), filtered['eta_dp'])
    elif 'revised_eta' in filtered.columns:
        filtered['eta_for_filter'] = filtered['revised_eta']
    else:
        filtered['eta_for_filter'] = filtered['eta_dp']

    # **CRITICAL FIX**: Use start_date and end_date from parse_time_period()
    date_mask = (filtered['eta_for_filter'] >= start_date) & (filtered['eta_for_filter'] <= end_date)
    if 'ata_dp' in filtered.columns:
        date_mask &= filtered['ata_dp'].isna()

    arrivals = filtered[date_mask].copy()
    if arrivals.empty:
        return (
            f"No containers with ETA between {format_date_for_display(start_date)} and {format_date_for_display(end_date)} "
            f"for the requested port ('{port_code_query or port_name_query}')."
        )

    # ---------- 6) Build display ----------
    display_cols = ['container_number', 'po_number_multiple', 'discharge_port', 'revised_eta', 'eta_dp']
    # include the matching port column for context
    for pc in ['discharge_port', 'final_load_port'] + existing_port_cols:
        if pc in arrivals.columns:
            sample = arrivals[pc].astype(str).str.upper()
            if port_code_query:
                if sample.str.contains(re.escape(port_code_query), na=False).any():
                    display_cols.append(pc)
                    break
            else:
                first_word = (port_name_query.split()[0] if port_name_query else "")
                if first_word and sample.str.contains(re.escape(first_word), na=False).any():
                    display_cols.append(pc)
                    break
                if sample.notna().any():
                    display_cols.append(pc)
                    break

    # Always include eta_for_filter for sorting, then drop before returning
    display_cols = [c for c in (display_cols + ['eta_for_filter']) if c in arrivals.columns]

    result_df = arrivals[display_cols].sort_values('eta_for_filter').head(100).copy()

    # Format date columns
    for dcol in ['revised_eta', 'eta_dp', 'eta_for_filter']:
        if dcol in result_df.columns and pd.api.types.is_datetime64_any_dtype(result_df[dcol]):
            result_df[dcol] = result_df[dcol].dt.strftime('%Y-%m-%d')

    # Drop internal helper column from final output
    if 'eta_for_filter' in result_df.columns:
        result_df = result_df.drop(columns=['eta_for_filter'])

    result_data = result_df.where(pd.notnull(result_df), None)

    return result_data.to_dict(orient="records")


# 6️⃣ Keyword / fuzzy search
def lookup_keyword(query: str) -> str:
    """
    Perform a fuzzy keyword search across all shipment data fields.
    Input: Query can include any relevant keywords (container, PO, BL, port, etc.).
    Output: Table of matching rows with priority columns.
    If no match, prompts for different keywords.
    """
    df = _df()
    words = query.upper().split()
    mask = df.apply(lambda r: any(w in str(r).upper() for w in words), axis=1)
    hits = df[mask]
    if hits.empty:
        return "No rows match the supplied keywords."

    priority = [
        "container_number", "po_number_multiple", "ocean_bl_no_multiple",
        "booking_number_multiple", "discharge_port", "eta_dp"
    ]
    cols = [c for c in priority if c in hits.columns][:5]
    return hits[cols].head(10).to_string(index=False)


# 7️⃣ PandasAI data analysis
def analyze_data_with_pandas(query: str) -> str:
    """
    Analyze shipping data using pandas based on a natural language query.
    Input: Ask any statistical or calculation question about the data.
    Output: Analysis result or summary.
    """
    df = _df()
    # Example: simple keyword-based logic
    if "average delay" in query.lower():
        if "delay_days" not in df.columns:
            df["delay_days"] = (pd.to_datetime(df["ata_dp"], errors="coerce") - pd.to_datetime(df["eta_dp"], errors="coerce")).dt.days
        avg_delay = df["delay_days"].mean()
        return f"Average delay is {avg_delay:.2f} days."
    elif "total containers" in query.lower():
        return f"Total containers: {len(df)}"
    else:
        return "Sorry, I can only answer questions about average delay or total containers right now."


# 8️⃣ Field information (generic) – the biggest function in the original script
def get_field_info(query: str) -> str:
    """
    Retrieve detailed information for a container or summarize a specific field.
    Input: Query can be a container number or mention a field (port, vessel, carrier, date, location).
    Output: For containers, shows all non-null fields. For fields, summarizes top values or date range.
    Uses fuzzy matching for ambiguous queries.
    """

    df = _df()
    container_no = extract_container_number(query)

    # a) If a container number is present → show its fields
    if container_no:
        rows = df[df["container_number"].astype(str).str.replace(" ", "").str.upper() == clean_container_number(container_no)]
        if rows.empty:
            return f"No data for container {container_no}."

        row = rows.iloc[0]
        # Show only non‑null fields
        lines = []
        for col, val in row.items():
            if pd.notnull(val) and str(val).strip() not in {"nan", ""}:
                if pd.api.types.is_datetime64_dtype(df[col]) or isinstance(val, pd.Timestamp):
                    val = val.strftime("%Y-%m-%d")
                lines.append(f"{col.replace('_', ' ').title()}: {val}")

        return f"Information for container {container_no}:\n" + "\n".join(lines[:15])


    # b) No container – try to infer which *type* of field the user wants
    field_patterns = {
        "port": r"\b(load\s*port|final\s*load\s*port|discharge\s*port|last\s*cy\s*location|place\s*of\s*receipt|place\s*of\s*delivery|final\s*destination)\b",
        "vessel": r"\b(first\s*vessel\s*code|first\s*vessel\s*name|first\s*voyage\s*code|final\s*vessel\s*code|final\s*vessel\s*name|final\s*voyage\s*code)\b",
        "carrier": r"\b(final\s*carrier\s*code|final\s*carrier\s*scac\s*code|final\s*carrier\s*name|true\s*carrier\s*code|true\s*carrier\s*scac\s*code)\b",
        "date": r"\b(etd_lp|etd_flp|eta_dp|eta_fd|revised_eta|predictive_eta|atd_lp|ata_flp|atd_flp|ata_dp|revised_eta_fd|predictive_eta_fd|cargo_received_date_multiple)\b",
        "location": r"\b(final_destination|carrier_vehicle_load_lcn|vehicle_departure_lcn|vehicle_arrival_lcn|carrier_vehicle_unload_lcn|out_gate_location|equipment_arrival_at_last_lcn|out_gate_at_last_cy_lcn|delivery_location_to_consignee|empty_container_return_lcn)\b",
    }

    field_type = None
    for ftype, pattern in field_patterns.items():
        if re.search(pattern, query, re.IGNORECASE):
            field_type = ftype
            break

    # c) Determine column list based on detected field type or fuzzy match
    if field_type == "port":
        cols = [c for c in ["load_port", "final_load_port", "discharge_port",
                            "last_cy_location", "place_of_receipt",
                            "place_of_delivery", "final_destination"]
                if c in df.columns]
    elif field_type == "vessel":
        cols = [c for c in ["first_vessel_code", "first_vessel_name",
                            "first_voyage_code", "final_vessel_code",
                            "final_vessel_name", "final_voyage_code"]
                if c in df.columns]
    elif field_type == "carrier":
        cols = [c for c in ["final_carrier_code", "final_carrier_scac_code",
                            "final_carrier_name", "true_carrier_code",
                            "true_carrier_scac_code"]
                if c in df.columns]
    elif field_type == "date":
        cols = [c for c in ["etd_lp", "etd_flp", "eta_dp", "eta_fd",
                            "revised_eta", "predictive_eta", "atd_lp",
                            "ata_flp", "atd_flp", "ata_dp", "revised_eta_fd",
                            "predictive_eta_fd", "cargo_received_date_multiple"]
                if c in df.columns]
    elif field_type == "location":
        cols = [c for c in ["final_destination","carrier_vehicle_load_lcn", "vehicle_departure_lcn",
                            "vehicle_arrival_lcn", "carrier_vehicle_unload_lcn",
                            "out_gate_location", "equipment_arrival_at_last_lcn",
                            "out_gate_at_last_cy_lcn", "delivery_location_to_consignee",
                            "empty_container_return_lcn"]
                if c in df.columns]
    else:
        # No explicit pattern → fuzzy‑match query words against column names
        words = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_]+\b", query)
                 if w.lower() not in {"what", "is", "the", "for", "of", "in", "on", "at", "by", "to"}]
        matches = []
        for w in words:
            if len(w) > 2:
                best = process.extractOne(w, df.columns.tolist())
                if best and best[1] > 70:        # 0‑100 fuzzy score
                    matches.append(best[0])
        cols = list(set(matches))

    if not cols:
        return "I couldn't determine which field you need. Please mention a port, vessel, carrier, date, etc."

    # d) Show a concise summary (top 5 values / date range)
    out_lines = []
    for c in cols[:3]:          # limit to three columns for brevity
        if pd.api.types.is_datetime64_dtype(df[c]):
            non_null = df[c].dropna()
            if not non_null.empty:
                out_lines.append(f"{c.replace('_', ' ').title()}:")
                out_lines.append(f"  earliest = {non_null.min().date()}, latest = {non_null.max().date()}, count = {non_null.count()}")
        else:
            vc = df[c].value_counts().head(5)
            if not vc.empty:
                out_lines.append(f"{c.replace('_', ' ').title()} (top 5 values):")
                out_lines.append("\n".join([f"  {val}: {cnt}" for val, cnt in vc.items()]))

    return "\n".join(out_lines) if out_lines else "No data available for the requested field."


# 9️⃣ Vessel Info (tiny helper that was separate in the original script)
def get_vessel_info(input_str: str) -> str:
    """
    Get vessel details for a specific container.
    Input: Provide a valid container number (partial or full).
    Output: Vessel codes and names for the container.
    If not found, prompts for a valid container number.
    """
    container_no = extract_container_number(input_str)
    if not container_no:
        return "Please specify a valid container number."

    df = _df()
    df["container_number"] = df["container_number"].astype(str)

    # exact match → then contains → then prefix
    clean = clean_container_number(container_no)
    rows = df[df["container_number"].str.replace(" ", "").str.upper() == clean]

    if rows.empty:
        rows = df[df["container_number"].str.contains(container_no, case=False, na=False)]

    if rows.empty:
        return f"No data found for container {container_no}."

    row = rows.iloc[0]
    parts = []
    for col in ["first_vessel_code", "first_vessel_name",
                "final_vessel_code", "final_vessel_name"]:
        if col in row and pd.notnull(row[col]) and str(row[col]).strip():
            parts.append(f"{col.replace('_', ' ').title()}: {row[col]}")

    if not parts:
        return f"No vessel information stored for container {container_no}."

    return f"Vessel information for container {container_no}:\n" + "\n".join(parts)


# 10️⃣ Upcoming PO's (by ETD window, ATA null)

def get_upcoming_pos(query: str, consignee_code: str = None) -> str:
    """
    List PO's scheduled to ship in the next X days (ETA window, not yet arrived).

    Args:
        query: Natural language query (may include days, location, PO token, consignee name, etc.)
        consignee_code: Optional exact consignee code or comma-separated codes to force filtering
                       by consignee code (e.g. "0000866" or "0045831,0043881")

    Returns:
        - list[dict] rows with PO / ETA / destination / consignee, OR
        - short string message (e.g. "No upcoming POs..." or "Yes — PO ... arrives on ...")

    Notes:
        - PO matching is prefix-insensitive (PO123, 123, PO-123 all match).
        - If query explicitly asks about a PO (e.g., "is 5302816722 arriving...")
          function returns a short yes/no string + details.
    """
    query = (query or "").strip()
    query_upper = query.upper()

    # Parse time period using centralized helper
    start_date, end_date, period_desc = parse_time_period(query)

    try:
        logger.info(
            f"[get_upcoming_pos] Period: {period_desc}, "
            f"Dates: {format_date_for_display(start_date)} to "
            f"{format_date_for_display(end_date)}"
        )
    except Exception:
        pass

    df = _df()

    # DATE selection using revised_eta / eta_dp ONLY
    date_priority = [c for c in ['revised_eta', 'eta_dp'] if c in df.columns]
    if not date_priority:
        return "No ETA columns (revised_eta / eta_dp) found in the data to compute upcoming arrivals."

    parse_cols = date_priority.copy()
    if 'ata_dp' in df.columns:
        parse_cols.append('ata_dp')
    df = ensure_datetime(df, parse_cols)

    # **CRITICAL FIX**: per-row ETA: prefer revised_eta over eta_dp, only when ata_dp is null
    if 'revised_eta' in df.columns and 'eta_dp' in df.columns:
        df['eta_for_filter'] = df['revised_eta'].where(
            df['revised_eta'].notna(),
            df['eta_dp']
        )
    elif 'revised_eta' in df.columns:
        df['eta_for_filter'] = df['revised_eta']
    else:
        df['eta_for_filter'] = df['eta_dp']

    # Build mask: eta_for_filter within window and ata_dp is null (not arrived)
    mask = (df['eta_for_filter'] >= start_date) & (df['eta_for_filter'] <= end_date)
    if 'ata_dp' in df.columns:
        mask &= df['ata_dp'].isna()  # **CRITICAL**: Only include rows where ata_dp is null

    candidate_df = df[mask].copy()

    if candidate_df.empty:
        return f"No upcoming POs in the {period_desc}."

    # Helper: normalize PO forms (prefix-insensitive)
    def normalize_po_text(s: str) -> str:
        """Normalize PO text for comparison."""
        if s is None:
            return ""
        s = str(s).upper()
        s = re.sub(r'[^A-Z0-9]', '', s)  # Remove non-alphanum
        if s.startswith("PO"):
            s = s[2:]
        return s.lstrip('0')

    # 1) If caller provided consignee_code (exact code(s)), filter by those code(s)
    if consignee_code:
        cc = str(consignee_code).strip().upper()
        cc_list = [c.strip().upper() for c in cc.split(',') if c.strip()]
        cc_set = set(cc_list)

        if 'consignee_code_multiple' in candidate_df.columns:
            def row_has_code(cell):
                """Check if cell contains any of the target consignee codes."""
                if pd.isna(cell):
                    return False

                parts = {
                    p.strip().upper()
                    for p in re.split(r',\s*', str(cell))
                    if p.strip()
                }

                for part in parts:
                    # Direct match
                    if part in cc_set:
                        return True

                    # Extract code from "NAME(CODE)" format
                    m = re.search(r'\(([A-Z0-9\- ]+)\)\s*$', part)
                    if m:
                        code = m.group(1).strip().upper()
                        if (code in cc_set or
                            code.lstrip('0') in cc_set or
                            code in {c.lstrip('0') for c in cc_set}):
                            return True

                return False

            candidate_df = candidate_df[
                candidate_df['consignee_code_multiple'].apply(row_has_code)
            ].copy()

            if candidate_df.empty:
                return (
                    f"No upcoming POs for consignee code(s) "
                    f"{', '.join(cc_list)} in the {period_desc}."
                )
        else:
            return "Dataset does not contain 'consignee_code_multiple' to filter by consignee code."

    # 2) Detect if query mentions a consignee name
    if 'consignee_code_multiple' in candidate_df.columns:
        try:
            all_names = candidate_df['consignee_code_multiple'].dropna().astype(str).unique().tolist()
            q_up = query.upper()

            consignee_name_filter = None
            for name in all_names:
                clean_name = re.sub(r"\([^)]*\)", "", name).strip().upper()
                if clean_name and clean_name in q_up:
                    consignee_name_filter = clean_name
                    break

            if consignee_name_filter:
                candidate_df = candidate_df[
                    candidate_df['consignee_code_multiple'].astype(str).str.upper().str.contains(
                        consignee_name_filter,
                        na=False
                    )
                ].copy()

                if candidate_df.empty:
                    return f"No upcoming POs found for consignee '{consignee_name_filter}'."
        except Exception as e:
            logger.debug(f"Error in consignee name detection: {e}")

    # 3) Detect port/location tokens in query
    port_cols = [
        c for c in ['discharge_port', 'vehicle_arrival_lcn', 'final_destination']
        if c in candidate_df.columns
    ]

    if port_cols:
        location_mask = pd.Series(False, index=candidate_df.index, dtype=bool)
        location_found = False
        location_name = None

        # Pattern 1: Port codes in parentheses like (USLAX)
        paren_match = re.search(r'\(([A-Z0-9]{3,6})\)', query_upper)
        if paren_match:
            tok = paren_match.group(1)
            for col in port_cols:
                location_mask |= candidate_df[col].astype(str).str.upper().str.contains(
                    rf'\({re.escape(tok)}\)',
                    na=False
                )
            if location_mask.any():
                location_found = True
                location_name = tok

        # Pattern 2: City names with prepositions
        if not location_found:
            city_patterns = [
                r'\b(?:AT|IN|TO)\s+([A-Z][A-Za-z\s\.\-]{3,}?)(?:\s+IN\s+|\s+NEXT\s+|,|\s*$)',
                r'\b(LOS\s+ANGELES|LONG\s+BEACH|SINGAPORE|ROTTERDAM|HONG\s+KONG)\b'
            ]

            for pattern in city_patterns:
                city_match = re.search(pattern, query, re.IGNORECASE)
                if city_match:
                    city = city_match.group(1).strip().upper()
                    # Clean up the matched city name
                    city = re.sub(r'\s+(IN\s+)?NEXT.*$', '', city, flags=re.IGNORECASE).strip()

                    if city and len(city) > 2:
                        location_name = city
                        for col in port_cols:
                            location_mask |= candidate_df[col].astype(str).str.upper().str.contains(
                                re.escape(city),
                                na=False
                            )

                        if location_mask.any():
                            location_found = True
                            break

        # Apply location filter if found
        if location_found:
            candidate_df = candidate_df[location_mask].copy()
            if candidate_df.empty:
                return (
                    f"No upcoming POs arriving at {location_name} "
                    f"in the {period_desc}."
                )

    # **CRITICAL FIX**: 4) PO-specific queries - ONLY extract POs from query text,
    # NOT from consignee_code

    # Build a set of known consignee codes to exclude from PO matching
    known_consignee_codes = set()
    if consignee_code:
        known_consignee_codes.update([
            c.strip().upper()
            for c in str(consignee_code).split(',')
            if c.strip()
        ])

    if 'consignee_code_multiple' in candidate_df.columns:
        try:
            for raw in candidate_df['consignee_code_multiple'].dropna().astype(str).tolist():
                for part in re.split(r',\s*', raw):
                    m = re.search(r'\(([A-Z0-9\- ]+)\)\s*$', part.strip().upper())
                    if m:
                        code = m.group(1).strip().upper()
                        known_consignee_codes.add(code)
                        known_consignee_codes.add(code.lstrip('0'))
        except Exception as e:
            logger.debug(f"Error extracting consignee codes: {e}")

    try:
        logger.info(
            f"[get_upcoming_pos] Known consignee codes to exclude from PO matching: "
            f"{known_consignee_codes}"
        )
    except Exception:
        pass

    # Extract PO tokens ONLY from query text, excluding consignee codes
    po_tokens = []

    # Pattern 1: Explicit "PO" prefix (e.g., "PO5302816722", "PO-6300134648")
    explicit_pos = re.findall(r'\bPO[-]?(\d{5,})\b', query_upper)
    po_tokens.extend(explicit_pos)

    # Pattern 2: Numeric tokens (only if query explicitly mentions "PO" or "purchase order")
    if re.search(r'\b(PO|PURCHASE\s+ORDER)\b', query_upper):
        # Only extract numeric tokens if query context suggests POs
        numeric_tokens = re.findall(r'\b(\d{5,})\b', query_upper)

        # Filter out known consignee codes
        for token in numeric_tokens:
            if (token not in known_consignee_codes and
                token.lstrip('0') not in known_consignee_codes):
                po_tokens.append(token)

    try:
        logger.info(f"[get_upcoming_pos] Extracted PO tokens from query: {po_tokens}")
    except Exception:
        pass

    # Handle PO-specific queries
    if po_tokens:
        # Normalize PO tokens (remove leading zeros)
        po_norms = {
            pn.lstrip('0')
            for pn in [re.sub(r'[^0-9]', '', p) for p in po_tokens]
            if pn
        }

        try:
            logger.info(f"[get_upcoming_pos] Normalized PO tokens: {po_norms}")
        except Exception:
            pass

        if 'po_number_multiple' in candidate_df.columns:
            def row_norms(cell):
                """Extract normalized PO numbers from a cell."""
                if pd.isna(cell):
                    return set()
                parts = [
                    p.strip()
                    for p in re.split(r',\s*', str(cell))
                    if p.strip()
                ]
                return {normalize_po_text(p) for p in parts if p}

            # Find matching rows
            matches = []
            for idx, row in candidate_df.iterrows():
                norms = row_norms(row.get('po_number_multiple'))
                if norms & po_norms:  # Set intersection
                    matches.append(row)

            if matches:
                # Check if query is a yes/no question about a specific PO
                is_yes_no_query = (
                    re.search(r'^\s*IS\b', query_upper) or
                    (re.search(r'\bARRIVING\b', query_upper) and '?' in query)
                )

                if is_yes_no_query:
                    # Return concise yes/no answer
                    first = matches[0]
                    etd_field = next(
                        (c for c in date_priority if c in candidate_df.columns),
                        None
                    )
                    etd_val = first.get(etd_field) if etd_field else None
                    display_etd = (
                        pd.to_datetime(etd_val).strftime('%Y-%m-%d')
                        if pd.notna(etd_val)
                        else None
                    )
                    po_display = first.get('po_number_multiple', None)
                    port_display = first.get('discharge_port', None)

                    return (
                        f"Yes — PO {po_tokens[0]} is scheduled "
                        f"(ETA: {display_etd}) to {port_display}. "
                        f"Row PO(s): {po_display}"
                    )
                else:
                    # Return detailed records
                    matches_df = pd.DataFrame(matches)
                    out_cols = (
                        ['po_number_multiple'] +
                        date_priority +
                        ["container_number", 'discharge_port', 'final_destination',
                         'consignee_code_multiple']
                    )
                    out_cols = [c for c in out_cols if c in matches_df.columns]

                    # Format dates
                    for d in date_priority:
                        if (d in matches_df.columns and
                            pd.api.types.is_datetime64_any_dtype(matches_df[d])):
                            matches_df[d] = matches_df[d].dt.strftime('%Y-%m-%d')

                    return matches_df[out_cols].drop_duplicates().to_dict(orient='records')
            else:
                # No matches found for the specific PO
                return (
                    f"No — PO {po_tokens[0]} is not scheduled to ship to the "
                    f"requested location/consignee in the {period_desc}."
                )

    # If NO PO tokens found in query, return all upcoming POs for the consignee
    try:
        logger.info("[get_upcoming_pos] No PO tokens found in query, returning all upcoming POs")
    except Exception:
        pass

    # Final output - return all upcoming POs
    po_col = (
        "po_number_multiple" if "po_number_multiple" in candidate_df.columns
        else ("po_number" if "po_number" in candidate_df.columns else None)
    )

    out_cols = []
    if po_col:
        out_cols.append(po_col)

    # Add date columns
    out_cols += [c for c in date_priority if c in candidate_df.columns]

    # Add additional columns
    out_cols += [
        c for c in [
            'container_number',
            'discharge_port',
            'final_destination',
            'consignee_code_multiple'
        ]
        if c in candidate_df.columns
    ]

    # Remove duplicates while preserving order
    out_cols = list(dict.fromkeys(out_cols))

    # Sort and limit results
    result_df = (
        candidate_df[out_cols]
        .drop_duplicates()
        .sort_values(by=date_priority[0])
        .head(200)
        .copy()
    )

    # Format date columns
    for d in date_priority:
        if (d in result_df.columns and
            pd.api.types.is_datetime64_any_dtype(result_df[d])):
            result_df[d] = result_df[d].dt.strftime('%Y-%m-%d')

    return result_df.where(pd.notnull(result_df), None).to_dict(orient='records')


# 11️⃣ Delayed PO's (complex ETA / milestone logic)
def get_delayed_pos(question: str = None, consignee_code: str = None, **kwargs) -> str:
    """
    Find delayed POs with ETA/ATA difference logic.
    Supports:
      - Numeric filters (less than, more than, 1–3 days, etc.)
      - Consignee filtering (by code or name)
      - Location filtering (port name or code, fuzzy matching)
    """

    query = (question or "")
    df = _df()

    # Validate required columns
    po_col = "po_number_multiple" if "po_number_multiple" in df.columns else "po_number"
    if po_col not in df.columns:
        return "PO column missing from data."

    date_cols = ["eta_dp", "ata_dp", "predictive_eta_fd", "revised_eta_fd", "eta_fd"]
    df = ensure_datetime(df, [c for c in date_cols if c in df.columns])

    # Apply consignee filter
    if consignee_code and "consignee_code_multiple" in df.columns:
        codes = [c.strip() for c in str(consignee_code).split(",") if c.strip()]
        mask = pd.Series(False, index=df.index)
        for c in codes:
            mask |= df["consignee_code_multiple"].astype(str).str.contains(c)
        df = df[mask].copy()
    if df.empty:
        return "No PO records found for provided consignee codes."

    # Detect consignee name in question
    consignee_name_filter = None
    if "consignee_code_multiple" in df.columns:
        all_names = df["consignee_code_multiple"].dropna().astype(str).unique().tolist()
        q_up = query.upper()
        for name in all_names:
            clean_name = re.sub(r"\([^)]*\)", "", name).strip().upper()
            if clean_name and clean_name in q_up:
                consignee_name_filter = clean_name
                break
        if consignee_name_filter:
            df = df[df["consignee_code_multiple"].astype(str).str.upper().str.contains(consignee_name_filter)]
            if df.empty:
                return f"No delayed POs found for consignee '{consignee_name_filter}'."

    # Compute delay_days
    if "ata_dp" in df.columns and "eta_dp" in df.columns:
        df["delay_days"] = (df["ata_dp"] - df["eta_dp"]).dt.days
    elif "predictive_eta_fd" in df.columns and "eta_fd" in df.columns:
        df["delay_days"] = (df["predictive_eta_fd"] - df["eta_fd"]).dt.days
    else:
        df["delay_days"] = pd.Series(0, index=df.index)

    df["delay_days"] = df["delay_days"].fillna(0).astype(int)

    arrived = df[df["ata_dp"].notna()].copy()
    if arrived.empty:
        return "No POs have arrived for your authorized consignees."

    # Location filter (code, name, or fuzzy)
    port_cols = [c for c in ["discharge_port", "final_destination", "place_of_delivery", "load_port"] if c in arrived.columns]

    def _extract_loc_code_and_name(q: str):
        q_up = (q or "").upper()
        m = re.search(r"\(([A-Z0-9]{3,6})\)", q_up)
        if m:
            return m.group(1), None

        cand_codes = set(re.findall(r"\b[A-Z0-9]{3,6}\b", q_up))
        if port_cols and cand_codes:
            known_codes = set()
            for c in port_cols:
                vals = arrived[c].dropna().astype(str).str.upper()
                known_codes |= set(re.findall(r"\(([A-Z0-9]{3,6})\)", " ".join(vals.tolist())))
            for code in cand_codes:
                if code in known_codes:
                    return code, None

        # Named port (at/in/on/from/to)
        m2_list = re.findall(r"(?:\b(?:ON|AT|IN|TO|FROM)\s+([A-Z][A-Z0-9\s\.,'\-]{2,}))", q_up)
        if m2_list:
            cand = max(m2_list, key=len).strip()
            cand = re.sub(r"(?:\d+\s*DAYS?|DELAY|LATE|BEHIND|ETA|BY).*", "", cand).strip()
            if cand:
                return None, cand

        # Fuzzy match fallback
        all_ports = set()
        for c in port_cols:
            vals = arrived[c].dropna().astype(str)
            vals = vals.str.replace(r"\([^)]*\)", "", regex=True).str.strip().str.upper().tolist()
            all_ports.update(vals)

        if all_ports:
            best = process.extractOne(q_up, all_ports, scorer=fuzz.token_set_ratio, score_cutoff=85)
            if best:
                return None, best[0]
        return None, None

    code, name = _extract_loc_code_and_name(query)

    if code or name:
        loc_mask = pd.Series(False, index=arrived.index)
        if code:
            for c in port_cols:
                loc_mask |= arrived[c].astype(str).str.upper().str.contains(rf"\({re.escape(code)}\)", na=False)
        else:
            tokens = [t for t in re.split(r"\W+", (name or "")) if len(t) >= 3]
            for c in port_cols:
                col_vals = arrived[c].astype(str).str.upper()
                cond = pd.Series(True, index=arrived.index)
                for t in tokens:
                    cond &= col_vals.str.contains(re.escape(t), na=False)
                loc_mask |= cond
        arrived = arrived[loc_mask].copy()
        if arrived.empty:
            where = f"{code or name}"
            return f"No delayed POs found at {where} for your authorized consignees."

    q = query.lower()

    # **CRITICAL FIX**: Add debug logging and ensure correct pattern matching
    try:
        logger.info(f"[get_delayed_pos] Processing query: {query}")
    except:
        pass

    # Delay day filters (order: range, <, >, X+, >=, ==, default >=7)
    range_match = re.search(r"(\d+)\s*[-–—]\s*(\d+)\s*days?", q)
    less_than = re.search(r"(?:less\s+than|under|below|<)\s*(\d+)\s*days?", q)
    more_than = re.search(r"(?:more\s+than|over|above|greater\s+than|>)\s*(\d+)\s*days?", q)  # Added 'greater than'
    plus_sign = re.search(r"\b(\d+)\s*\+\s*days?\b", q)
    at_least = re.search(r"(?:at\s+least|>=|minimum)\s*(\d+)\s*days?", q)
    exact = re.search(r"(?:exactly|by|of|in)\s+(\d+)\s+days?", q)

    # Log which pattern matched
    try:
        if range_match:
            logger.info(f"[get_delayed_pos] Matched range pattern: {range_match.groups()}")
        elif less_than:
            logger.info(f"[get_delayed_pos] Matched less_than pattern: {less_than.groups()}")
        elif more_than:
            logger.info(f"[get_delayed_pos] Matched more_than pattern: {more_than.groups()}")
        elif plus_sign:
            logger.info(f"[get_delayed_pos] Matched plus_sign pattern: {plus_sign.groups()}")
        elif at_least:
            logger.info(f"[get_delayed_pos] Matched at_least pattern: {at_least.groups()}")
        elif exact:
            logger.info(f"[get_delayed_pos] Matched exact pattern: {exact.groups()}")
        else:
            logger.info(f"[get_delayed_pos] No pattern matched, using default >= 7")
    except:
        pass

    if range_match:
        d1, d2 = int(range_match.group(1)), int(range_match.group(2))
        low, high = min(d1, d2), max(d1, d2)
        delayed = arrived[(arrived["delay_days"] >= low) & (arrived["delay_days"] <= high)]
        try:
            logger.info(f"[get_delayed_pos] Range filter: {low} <= delay_days <= {high}, results: {len(delayed)}")
        except:
            pass
    elif less_than:
        d = int(less_than.group(1))
        delayed = arrived[(arrived["delay_days"] > 0) & (arrived["delay_days"] < d)]
        try:
            logger.info(f"[get_delayed_pos] Less than filter: 0 < delay_days < {d}, results: {len(delayed)}")
        except:
            pass
    elif more_than:
        d = int(more_than.group(1))
        # **CRITICAL**: Use strictly greater than (>) for "more than X days"
        delayed = arrived[arrived["delay_days"] > d]
        try:
            logger.info(f"[get_delayed_pos] More than filter: delay_days > {d}, results: {len(delayed)}")
            logger.info(f"[get_delayed_pos] Sample delay_days values: {arrived['delay_days'].value_counts().head(10).to_dict()}")
        except:
            pass
    elif plus_sign:
        d = int(plus_sign.group(1))
        delayed = arrived[arrived["delay_days"] >= d]
        try:
            logger.info(f"[get_delayed_pos] Plus sign filter: delay_days >= {d}, results: {len(delayed)}")
        except:
            pass
    elif at_least:
        d = int(at_least.group(1))
        delayed = arrived[arrived["delay_days"] >= d]
        try:
            logger.info(f"[get_delayed_pos] At least filter: delay_days >= {d}, results: {len(delayed)}")
        except:
            pass
    elif exact:
        d = int(exact.group(1))
        delayed = arrived[arrived["delay_days"] == d]
        try:
            logger.info(f"[get_delayed_pos] Exact filter: delay_days == {d}, results: {len(delayed)}")
        except:
            pass
    else:
        delayed = arrived[arrived["delay_days"] >= 7]
        try:
            logger.info(f"[get_delayed_pos] Default filter: delay_days >= 7, results: {len(delayed)}")
        except:
            pass

    if delayed.empty:
        where = f" at {code or name}" if (code or name) else ""
        return f"No delayed POs found for your authorized consignees{where}."

    # Output formatting
    cols = [po_col, "container_number", "eta_dp", "ata_dp", "delay_days",
            "consignee_code_multiple", "discharge_port"]
    cols = [c for c in cols if c in delayed.columns]

    out = delayed[cols].sort_values("delay_days", ascending=False).head(100).copy()
    for dcol in ["eta_dp", "ata_dp"]:
        if dcol in out.columns and pd.api.types.is_datetime64_any_dtype(out[dcol]):
            out[dcol] = out[dcol].dt.strftime("%Y-%m-%d")

    return out.where(pd.notnull(out), None).to_dict(orient="records")


# 12️⃣ Containers arriving soon (ETA window & ATA null)
def get_containers_arriving_soon(query: str) -> str:
    """
    List containers arriving soon (ETA window, ATA is null) - now with consignee filtering.
    """
    m = re.search(r"(?:next|in|upcoming|within)\s+(\d{1,3})\s+days?", query, re.IGNORECASE)
    days = int(m.group(1)) if m else 7

    df = _df()  # This now automatically filters by consignee

    # transport-mode filter (if present)
    modes = extract_transport_modes(query)
    if modes and 'transport_mode' in df.columns:
        df = df[df['transport_mode'].astype(str).str.lower().apply(lambda s: any(m in s for m in modes))]

    # choose per-row ETA preference: revised_eta > eta_dp
    date_priority = [c for c in ['revised_eta', 'eta_dp'] if c in df.columns]
    if not date_priority:
        return "No ETA columns (revised_eta / eta_dp) found in the data."

    parse_cols = date_priority.copy()
    if 'ata_dp' in df.columns:
        parse_cols.append('ata_dp')
    df = ensure_datetime(df, parse_cols)

    # create eta_for_filter per-row
    if 'revised_eta' in df.columns and 'eta_dp' in df.columns:
        df['eta_for_filter'] = df['revised_eta'].where(df['revised_eta'].notna(), df['eta_dp'])
    elif 'revised_eta' in df.columns:
        df['eta_for_filter'] = df['revised_eta']
    else:
        df['eta_for_filter'] = df['eta_dp']

    today = pd.Timestamp.today().normalize()
    future = today + pd.Timedelta(days=days)

    # Build mask safely (eta_for_filter within window) and exclude already-arrived (ata_dp not null)
    mask = df['eta_for_filter'].notna() & (df['eta_for_filter'] >= today) & (df['eta_for_filter'] <= future)
    if 'ata_dp' in df.columns:
        mask &= df['ata_dp'].isna()

    upcoming = df[mask].copy()
    if upcoming.empty:
        return f"No containers arriving in the next {days} days for your authorized consignees."

    # Prepare output
    cols = ['container_number', 'discharge_port', 'po_number_multiple', 'eta_for_filter']
    cols = [c for c in cols if c in upcoming.columns]
    out = upcoming[cols].sort_values('eta_for_filter').head(50).copy()

    # format eta_for_filter
    if 'eta_for_filter' in out.columns and pd.api.types.is_datetime64_any_dtype(out['eta_for_filter']):
        out['eta_for_filter'] = out['eta_for_filter'].dt.strftime('%Y-%m-%d')

    # rename eta_for_filter back to a friendly column name for output (keep original names if needed)
    out = out.rename(columns={'eta_for_filter': 'eta'})

    return out.where(pd.notnull(out), None).to_dict(orient='records')


def check_arrival_status(input_str: str) -> str:
    """
    Check if a container or PO has arrived - now with consignee filtering.
    """
    from datetime import datetime

    # Extract container number or PO number from input
    container_match = re.search(r'([A-Z]{4}\d{7})', input_str)
    po_match = re.search(r'(?:po|purchase order)\s*[:\s]*([A-Z0-9]+)', input_str, re.IGNORECASE)

    df = _df()  # This automatically filters by consignee
    df = ensure_datetime(df, ["ata_dp", "derived_ata_dp", "eta_dp"])
    today = datetime.now().date()

    if container_match:
        container_number = container_match.group(1)
        # Search for container in authorized data
        container_df = df[df['container_number'] == container_number]

        if container_df.empty:
            return f"No data found for container {container_number} or you are not authorized to access this container."

        # Get the first record
        record = container_df.iloc[0]
        discharge_port = record.get('discharge_port', 'Unknown Port')

        # Check ata_dp first
        if pd.notnull(record['ata_dp']):
            ata_date = (
                record['ata_dp'].date()
                if hasattr(record['ata_dp'], 'date')
                else record['ata_dp']
            )
            return f"Container <con>{container_number}</con> reached on {ata_date} at {discharge_port} discharge port."

        # If ata_dp is null, check derived_ata_dp
        elif pd.notnull(record.get('derived_ata_dp')):
            derived_ata = (
                record['derived_ata_dp'].date()
                if hasattr(record['derived_ata_dp'], 'date')
                else record['derived_ata_dp']
            )
            if derived_ata <= today:
                return f"Container <con>{container_number}</con> reached on {derived_ata} at {discharge_port} discharge port."
            else:
                return f"Container <con>{container_number}</con> is on the water, expected at {discharge_port} discharge port on {derived_ata}."
        else:
            return f"Container <con>{container_number}</con> is on the water, discharge port: {discharge_port}."

    elif po_match:
        po_number = po_match.group(1)
        # Search for PO in authorized data
        po_col = "po_number_multiple" if "po_number_multiple" in df.columns else "po_number"
        po_df = df[df[po_col].astype(str).str.contains(po_number, case=False, na=False)]

        if po_df.empty:
            return f"No data found for PO {po_number} or you are not authorized to access this PO."

        # For PO, we might have multiple containers, so let's handle the most recent or relevant one
        if len(po_df) > 1:
            po_df = po_df.sort_values(['etd_lp', 'eta_dp'], ascending=[False, False])

        record = po_df.iloc[0]
        discharge_port = record.get('discharge_port', 'Unknown Port')

        # Check ata_dp first
        if pd.notnull(record['ata_dp']):
            ata_date = (
                record['ata_dp'].date()
                if hasattr(record['ata_dp'], 'date')
                else record['ata_dp']
            )
            return f"PO {po_number} reached on {ata_date} at {discharge_port} discharge port."

        # If ata_dp is null, check derived_ata_dp
        elif pd.notnull(record.get('derived_ata_dp')):
            derived_ata = (
                record['derived_ata_dp'].date()
                if hasattr(record['derived_ata_dp'], 'date')
                else record['derived_ata_dp']
            )
            if derived_ata <= today:
                return f"PO {po_number} reached on {derived_ata} at {discharge_port} discharge port."
            else:
                return f"PO {po_number} is on the water, expected at {discharge_port} discharge port on {derived_ata}."
        else:
            return f"PO {po_number} is on the water, discharge port: {discharge_port}."

    else:
        return "Please provide a valid container number or PO number to check arrival status."


def _normalize_agent_timephrase_for_week(query: str) -> str:
    """
    Agent sometimes rewrites user intent "this week" into: 'consignee_code=XXXX next 7 days departing'.
    This helper converts ONLY that agent-shaped input back into 'this week' so parse_time_period()
    applies calendar week (Mon–Sun).

    Guardrails (to avoid impacting other queries):
    - Require 'consignee_code=' prefix
    - Require literal 'next 7 days'
    - Require 'departing' or 'departed' keyword
    - Do NOT change if query already contains explicit week/month/range/day terms
    """
    q = (query or "").strip()
    if not q:
        return q

    q_low = q.lower()

    # If the query already contains an explicit calendar phrase or explicit range/date, do nothing
    if re.search(r"\b(this|current|next|last|previous)\s+(week|wk|month)\b", q_low):
        return q
    if re.search(r"\b(today|tomorrow|yesterday)\b", q_low):
        return q
    if re.search(r"\b(from|between)\b", q_low):
        return q
    if re.search(r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}\b|\b\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2}\b", q_low):
        return q

    # Only rewrite the known agent-shape
    if (
        re.search(r"\bconsignee_code\s*=\s*\d{5,10}\b", q_low)
        and re.search(r"\bnext\s+7\s+days\b", q_low)
        and re.search(r"\bdepart(?:ing|ed)\b", q_low)
    ):
        # Replace only the time phrase; keep the rest intact
        q2 = re.sub(r"\bnext\s+7\s+days\b", "this week", q, flags=re.IGNORECASE)
        # Normalize whitespace
        q2 = re.sub(r"\s+", " ", q2).strip()
        return q2

    return q
def _strip_consignee_noise(text: str) -> str:
    """
    Remove agent-injected consignee phrases that must NOT become part of port names.
    Examples removed:
      - "for consignee 0000866"
      - "consignee code 0000866"
      - "consignee_code=0000866"
    """
    s = (text or "").strip()
    if not s:
        return s

    s = re.sub(r"\bconsignee_code\s*=\s*\d{5,10}\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bconsignee\s*code\s*[:=]?\s*\d{5,10}\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bfor\s+consignee\s+\d{5,10}\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def get_containers_departing_from_load_port(query: str) -> str:
    """
    Get containers/POs/OBLs scheduled to depart from specific load ports in upcoming time periods.

    Supports:
    - Container queries: "containers will depart from SHANGHAI in next 5 days"
    - PO queries: "POs leaving from load port QINGDAO this week"
    - OBL queries: "BLs scheduled to leave NINGBO tomorrow"
    - Mixed queries: "show upcoming shipments from load port BUSAN in next 7 days"
    - Transport mode filtering: "containers departing by sea from HONG KONG"
    - Hot container filtering: "hot containers scheduled to leave SHANGHAI next week"

    Uses ETD_LP (Estimated Time of Departure from Load Port).
    Automatically excludes containers that have already departed (atd_lp is not null).
    Returns list[dict] with container/PO/OBL info and departure details.
    """

    # Fix agent-injected time phrase issue
    query = _normalize_agent_timephrase_for_week(query)

    # Strip consignee noise so port extraction doesn't capture "FOR CONSIGNEE 0000866"
    query_raw = (query or "").strip()
    query = _strip_consignee_noise(query_raw)

    query_upper = query.upper()
    query_lower = query.lower()

    # ========== 1) EXTRACT IDENTIFIERS (Container/PO/OBL) ==========
    # Remove consignee code mentions before extraction to avoid false positives
    query_for_extraction = query
    query_for_extraction = re.sub(
        r'\b(?:for\s+)?(?:consignee|user)(?:\s*[_\-]?\s*code)?\s*[:=]?\s*\d{7}\b',
        '',
        query_for_extraction,
        flags=re.IGNORECASE,
    )

    container_no = extract_container_number(query_for_extraction)
    po_no = extract_po_number(query_for_extraction)
    obl_no = None
    try:
        obl_no = extract_ocean_bl_number(query_for_extraction)
    except Exception:
        pass

    try:
        logger.info(f"[get_containers_departing_from_load_port] Extracted: container={container_no}, po={po_no}, obl={obl_no}")
    except:
        pass

    # ========== 2) EXTRACT LOAD PORT ==========
    load_port = None

    # List of time-related words to exclude from port name
    time_words = r"(?:tomorrow|today|yesterday|next|this|within|in|by|last|week|month|days?)"

    # Pattern 1: "from load port PORTNAME" or "from PORTNAME"
    # Stop before time-related words, "for", "in", "next", etc.
    match = re.search(
        rf"from\s+(?:load\s+port\s+)?([A-Za-z0-9\s,\-\(\)]+?)"
        rf"(?=\s+for\b|\s+in\s+|\s+next\s+|\s+this\s+|\s+within\s+|\s+on\s+|\s+by\s+|\s+{time_words}\b|[\?\.\,]|$)",
        query,
        re.IGNORECASE,
    )
    if match:
        cand = match.group(1).strip()
        # Remove any trailing time words that might have been captured
        cand = re.sub(rf"\s+(?:tomorrow|today|yesterday|next|this|within|in|by)\b.*$", "", cand, flags=re.IGNORECASE).strip()
        # Exclude common noise words
        if cand and cand.upper() not in [
            "CONSIGNEE", "FOR", "IN", "NEXT", "THE", "DAYS", "THIS", "WEEK", "SEA", "AIR", "ROAD", "ON",
            "TOMORROW", "TODAY", "YESTERDAY", "WITHIN", "BY"
        ]:
            load_port = cand.upper()

    # Pattern 2: "load port PORTNAME"
    if not load_port:
        match = re.search(
            rf"load\s+port\s+([A-Za-z0-9\s,\-\(\)]+?)"
            rf"(?=\s+for\b|\s+in\s+|\s+next\s+|\s+this\s+|\s+within\s+|\s+on\s+|\s+by\s+|\s+{time_words}\b|[\?\.\,]|$)",
            query,
            re.IGNORECASE,
        )
        if match:
            cand = match.group(1).strip()
            # Remove any trailing time words
            cand = re.sub(rf"\s+(?:tomorrow|today|yesterday|next|this|within|in|by)\b.*$", "", cand, flags=re.IGNORECASE).strip()
            if cand and cand.upper() not in ["TOMORROW", "TODAY", "YESTERDAY", "NEXT", "THIS", "WEEK"]:
                load_port = cand.upper()

    # Pattern 3: "will depart/leaving from PORTNAME"
    if not load_port:
        match = re.search(
            rf"(?:will\s+depart|departing|leaving|scheduled\s+to\s+leave)\s+(?:from\s+)?([A-Za-z0-9\s,\-\(\)]+?)"
            rf"(?=\s+for\b|\s+in\s+|\s+next\s+|\s+this\s+|\s+within\s+|\s+on\s+|\s+by\s+|\s+{time_words}\b|[\?\.\,]|$)",
            query,
            re.IGNORECASE,
        )
        if match:
            cand = match.group(1).strip()
            # Remove any trailing time words
            cand = re.sub(rf"\s+(?:tomorrow|today|yesterday|next|this|within|in|by)\b.*$", "", cand, flags=re.IGNORECASE).strip()
            if cand and cand.upper() not in [
                "CONSIGNEE", "FOR", "IN", "NEXT", "THE", "DAYS", "THIS", "WEEK", "SEA", "AIR", "ROAD", "ON",
                "TOMORROW", "TODAY", "YESTERDAY"
            ]:
                load_port = cand.upper()

    # Pattern 4: Port with code in parentheses like "SHANGHAI(CNSHA)"
    if not load_port:
        match = re.search(r'([A-Za-z0-9\s,\-]+?\([A-Z0-9\-]{3,6}\))', query, re.IGNORECASE)
        if match:
            load_port = re.sub(r'\s+', ' ', match.group(1).strip()).upper()

    # Clean up extracted port name - remove time-related words
    if load_port:
        load_port = re.sub(r'\s+', ' ', load_port).strip()
        # Final cleanup: remove time words if they somehow got captured
        load_port = re.sub(
            r'\b(?:tomorrow|today|yesterday|next|this|last|within|in|by|days?|weeks?|months?)\b',
            '',
            load_port,
            flags=re.IGNORECASE
        ).strip()
        # Remove extra whitespace
        load_port = re.sub(r'\s+', ' ', load_port).strip()

    try:
        logger.info(f"[get_containers_departing_from_load_port] Extracted load_port: '{load_port}'")
    except:
        pass

    # ========== 3) PARSE TIME WINDOW (FUTURE DATES) ==========
    start_date, end_date, period_desc = parse_time_period(query)

    # Ensure we're looking at future dates (for upcoming departures)
    today = pd.Timestamp.today().normalize()

    # If the parsed period is in the past, default to next 7 days
    if end_date < today:
        start_date = today
        end_date = today + pd.Timedelta(days=7)
        period_desc = "next 7 days (default)"

    try:
        logger.info(f"[get_containers_departing_from_load_port] Time window: {start_date} to {end_date} ({period_desc})")
    except:
        pass

    # ========== 4) LOAD AND FILTER DATA ==========
    df = _df()  # Respects consignee filtering

    try:
        logger.info(f"[get_containers_departing_from_load_port] After _df() consignee filter: {len(df)} rows")
        if hasattr(threading.current_thread(), 'consignee_codes'):
            codes = threading.current_thread().consignee_codes
            logger.info(f"[get_containers_departing_from_load_port] Thread consignee codes: {codes}")
    except:
        pass

    if df.empty:
        return "No data available for your authorized consignees."

    # ========== 5) APPLY IDENTIFIER FILTERS (Container/PO/OBL) ==========
    identifier_mask = pd.Series(True, index=df.index)
    identifier_type = None

    # Filter by Container (highest priority)
    if container_no:
        if 'container_number' not in df.columns:
            return f"Container column not found for container {container_no}."

        clean_cont = clean_container_number(container_no)
        cont_col_norm = df["container_number"].astype(str).str.replace(r'[^A-Z0-9]', '', regex=True)
        identifier_mask = cont_col_norm == clean_cont

        if not identifier_mask.any():
            identifier_mask = df["container_number"].astype(str).str.contains(container_no, case=False, na=False)

        identifier_type = "container"
        try:
            logger.info(f"[get_containers_departing_from_load_port] Container filter: {identifier_mask.sum()} rows matched")
        except:
            pass

    # Filter by PO
    elif po_no:
        po_col = "po_number_multiple" if "po_number_multiple" in df.columns else ("po_number" if "po_number" in df.columns else None)
        if not po_col:
            return f"PO column not found for PO {po_no}."

        po_norm = _normalize_po_token(po_no)
        identifier_mask = df[po_col].apply(lambda cell: _po_in_cell(cell, po_norm))
        identifier_type = "PO"

        try:
            logger.info(f"[get_containers_departing_from_load_port] PO filter: {identifier_mask.sum()} rows matched")
        except:
            pass

    # Filter by OBL
    elif obl_no:
        bl_col = _find_ocean_bl_col(df)
        if not bl_col:
            return f"Ocean BL column not found for OBL {obl_no}."

        bl_norm = _normalize_bl_token(obl_no)
        identifier_mask = df[bl_col].apply(lambda cell: _bl_in_cell(cell, bl_norm))
        identifier_type = "OBL"

        try:
            logger.info(f"[get_containers_departing_from_load_port] OBL filter: {identifier_mask.sum()} rows matched")
        except:
            pass

    # Apply identifier filter
    df = df[identifier_mask].copy()

    if df.empty:
        if identifier_type:
            return f"No data found for {identifier_type} {container_no or po_no or obl_no}."

    # ========== 5A) TRANSPORT MODE FILTER ==========
    modes = extract_transport_modes(query)
    if modes:
        try:
            logger.info(f"[get_containers_departing_from_load_port] Transport modes detected: {modes}")
        except:
            pass

        if 'transport_mode' in df.columns:
            mode_mask = df['transport_mode'].astype(str).str.lower().apply(lambda s: any(m in s for m in modes))
            df = df[mode_mask].copy()

            try:
                logger.info(f"[get_containers_departing_from_load_port] After transport mode filter: {len(df)} rows")
            except:
                pass

            if df.empty:
                mode_str = ", ".join(sorted(modes))
                desc = f"{identifier_type} {container_no or po_no or obl_no}" if identifier_type else "containers"
                return f"No {desc} scheduled to depart by {mode_str}."
        else:
            try:
                logger.warning(f"[get_containers_departing_from_load_port] transport_mode column not found, skipping mode filter")
            except:
                pass

    # ========== 5B) HOT CONTAINER FLAG FILTER ==========
    is_hot_query = bool(re.search(r'\bhot\b', query, re.IGNORECASE))
    if is_hot_query:
        try:
            logger.info(f"[get_containers_departing_from_load_port] Hot container filter requested")
        except:
            pass

        hot_flag_cols = [c for c in df.columns if 'hot_container_flag' in c.lower()]
        if not hot_flag_cols:
            hot_flag_cols = [c for c in df.columns if 'hot_container' in c.lower()]

        if hot_flag_cols:
            hot_col = hot_flag_cols[0]

            def _is_hot(v):
                if pd.isna(v):
                    return False
                return str(v).strip().upper() in {"Y", "YES", "TRUE", "1", "HOT"}

            hot_mask = df[hot_col].apply(_is_hot)
            df = df[hot_mask].copy()

            try:
                logger.info(f"[get_containers_departing_from_load_port] After hot flag filter: {len(df)} rows")
            except:
                pass

            if df.empty:
                desc = f"{identifier_type} {container_no or po_no or obl_no}" if identifier_type else "containers"
                return f"No hot {desc} scheduled to depart."
        else:
            try:
                logger.warning(f"[get_containers_departing_from_load_port] hot_container_flag column not found")
            except:
                pass

    # ========== 6) APPLY LOAD PORT FILTER ==========
    if load_port:
        try:
            logger.info(f"[get_containers_departing_from_load_port] Before port filter: {len(df)} rows")
            # Check if test containers exist before port filtering
            test_containers = ['MRKU0662058', 'TTNU4392192']
            for cont in test_containers:
                if 'container_number' in df.columns and cont in df['container_number'].values:
                    cont_row = df[df['container_number'] == cont].iloc[0]
                    logger.info(f"[get_containers_departing_from_load_port] BEFORE PORT FILTER - Found {cont}: load_port={cont_row.get('load_port')}, etd_lp={cont_row.get('etd_lp')}")
        except:
            pass

        if 'load_port' not in df.columns:
            return "Load port column not found in the dataset."

        # Normalize port names for matching
        def normalize_port_name(port_str):
            if pd.isna(port_str):
                return ""
            s = str(port_str).upper()
            s = re.sub(r'\([^)]*\)', '', s)  # Remove code in parentheses
            s = s.replace(',', ' ')
            s = re.sub(r'\s+', ' ', s).strip()
            return s

        def extract_port_code(port_str):
            if pd.isna(port_str):
                return None
            m = re.search(r'\(([A-Z0-9\-]+)\)', str(port_str).upper())
            return m.group(1) if m else None

        load_port_norm = normalize_port_name(load_port)
        df['_norm_port'] = df['load_port'].apply(normalize_port_name)
        df['_port_code'] = df['load_port'].apply(extract_port_code)

        # Try matching strategies
        user_code = None
        raw_up = load_port.upper()

        # Check if user provided just a port code
        m_code = re.search(r'\b([A-Z0-9\-]{3,6})\b$', raw_up)
        if m_code and load_port_norm == m_code.group(1):
            user_code = m_code.group(1)
        elif re.fullmatch(r'[A-Z0-9\-]{3,6}', load_port_norm.replace(' ', '')):
            user_code = load_port_norm.replace(' ', '')

        port_mask = pd.Series(False, index=df.index)

        # Strategy 1: Match by port code (highest priority)
        if user_code:
            port_mask = df['_port_code'].fillna('').str.upper() == user_code
            try:
                logger.info(f"[get_containers_departing_from_load_port] Code match '{user_code}': {port_mask.sum()} rows")
            except:
                pass

        # Strategy 2: Exact normalized name match
        if not port_mask.any():
            port_mask = df['_norm_port'] == load_port_norm
            try:
                logger.info(f"[get_containers_departing_from_load_port] Exact name match '{load_port_norm}': {port_mask.sum()} rows")
            except:
                pass

        # Strategy 3: Word-based contains - ALL user words must be present as word boundaries
        if not port_mask.any():
            user_words = [w for w in load_port_norm.split() if len(w) >= 2]
            if user_words:
                # Each word must be present as a word boundary (not substring)
                port_mask = pd.Series(True, index=df.index)
                for user_word in user_words:
                    word_match = df['_norm_port'].str.contains(rf'\b{re.escape(user_word)}\b', na=False, regex=True, case=False)
                    port_mask &= word_match

                try:
                    logger.info(f"[get_containers_departing_from_load_port] Word-based match (all words required as word boundaries) {user_words}: {port_mask.sum()} rows")
                except:
                    pass
            else:
                # Fallback: raw substring match for single short word
                port_mask = df['load_port'].astype(str).str.upper().str.contains(raw_up, na=False, regex=False)

        # Apply port filter
        df = df[port_mask].copy()

        # Clean up temporary columns
        df.drop(columns=['_norm_port', '_port_code'], inplace=True, errors='ignore')

        try:
            logger.info(f"[get_containers_departing_from_load_port] After port filter: {len(df)} rows matched")
            if len(df) > 0:
                sample_ports = df['load_port'].head(5).tolist()
                logger.info(f"[get_containers_departing_from_load_port] Sample matched ports: {sample_ports}")
                # Check for specific containers mentioned by user
                test_containers = ['MRKU0662058', 'TTNU4392192']
                for cont in test_containers:
                    if cont in df['container_number'].values:
                        cont_row = df[df['container_number'] == cont].iloc[0]
                        logger.info(f"[get_containers_departing_from_load_port] Found {cont}: load_port={cont_row.get('load_port')}, etd_lp={cont_row.get('etd_lp')}, atd_lp={cont_row.get('atd_lp')}")
        except:
            pass

        if df.empty:
            desc = f"{identifier_type} {container_no or po_no or obl_no}" if identifier_type else "containers"
            return f"No {desc} found scheduled to depart from load port {load_port}."

    # ========== 7) VALIDATE AND PARSE DATE COLUMNS ==========
    needed_cols = [c for c in ['etd_lp', 'atd_lp'] if c in df.columns]
    if not needed_cols:
        return "No departure date columns (etd_lp, atd_lp) found."

    # If ETD column doesn't exist, can't process upcoming departures
    if 'etd_lp' not in df.columns:
        return "ETD load port column (etd_lp) not found to determine scheduled departures."

    df = ensure_datetime(df, needed_cols)

    # ========== 8) FILTER BY ETD WINDOW (+ optional ATD exclusion) ==========
    try:
        logger.info(f"[get_containers_departing_from_load_port] Before date filter: {len(df)} rows, start_date={start_date}, end_date={end_date}, today={today}")
        if 'etd_lp' in df.columns:
            etd_sample = df[['container_number', 'etd_lp']].head(10)
            logger.info(f"[get_containers_departing_from_load_port] Sample ETD values:\n{etd_sample}")

            # Check for specific test containers
            test_containers = ['MRKU0662058', 'TTNU4392192']
            for cont in test_containers:
                if cont in df['container_number'].values:
                    cont_row = df[df['container_number'] == cont].iloc[0]
                    etd_val = cont_row.get('etd_lp')
                    etd_normalized = pd.to_datetime(etd_val).normalize() if pd.notna(etd_val) else None
                    logger.info(f"[get_containers_departing_from_load_port] TEST CONTAINER {cont}: etd_lp={etd_val}, normalized={etd_normalized}, in_range={etd_normalized >= start_date and etd_normalized <= end_date if etd_normalized else False}")
                else:
                    logger.info(f"[get_containers_departing_from_load_port] TEST CONTAINER {cont}: NOT FOUND in {len(df)} rows")
    except Exception as e:
        logger.error(f"[get_containers_departing_from_load_port] Error in debug logging: {e}", exc_info=True)

    # Base: ETD within the requested window (calendar-aware for 'this week')
    date_mask = (
        df['etd_lp'].notna()
        & (df['etd_lp'].dt.normalize() >= start_date)
        & (df['etd_lp'].dt.normalize() <= end_date)
    )

    try:
        logger.info(f"[get_containers_departing_from_load_port] After ETD date range filter: {date_mask.sum()} rows matched")
    except:
        pass

    # For calendar phrases like "this week", users often mean the full week window
    # even if some ETDs are earlier than today
    include_already_departed = False
    try:
        if start_date < today:
            if re.search(r"\b(this|current)\s+(week|wk)\b", query_lower) or re.search(r"\bthisweek\b", query_lower):
                include_already_departed = True
            # Explicit date ranges like "from 22/12/2025 to 28/12/2025" should also include the full window
            if re.search(r"\b(from|between)\b", query_lower):
                include_already_departed = True
    except Exception:
        include_already_departed = False

    if 'atd_lp' in df.columns and not include_already_departed:
        date_mask &= df['atd_lp'].isna()
        try:
            logger.info(f"[get_containers_departing_from_load_port] After ATD null filter: {date_mask.sum()} rows matched")
        except:
            pass

    results = df[date_mask].copy()

    if results.empty:
        desc = f"{identifier_type} {container_no or po_no or obl_no}" if identifier_type else "containers"
        port_desc = f" from load port {load_port}" if load_port else ""
        mode_desc = f" by {', '.join(sorted(modes))}" if modes else ""
        hot_desc = " (hot)" if is_hot_query else ""
        return f"No {desc}{hot_desc} scheduled to depart{port_desc}{mode_desc} in the {period_desc}."

    # ========== 9) SORT BY DEPARTURE DATE (EARLIEST FIRST) ==========
    results = results.sort_values('etd_lp', ascending=True)

    # ========== 10) PREPARE OUTPUT COLUMNS ==========
    output_cols = ['container_number', 'load_port', 'etd_lp']
    if 'atd_lp' in results.columns:
        output_cols.append('atd_lp')

    # Add PO/OBL columns if relevant
    if identifier_type == "PO" or not identifier_type:
        po_col = "po_number_multiple" if "po_number_multiple" in results.columns else ("po_number" if "po_number" in results.columns else None)
        if po_col:
            output_cols.append(po_col)

    if identifier_type == "OBL" or not identifier_type:
        bl_col = _find_ocean_bl_col(results)
        if bl_col:
            output_cols.append(bl_col)

    # Add additional context columns
    additional_cols = ['discharge_port', 'eta_dp', 'revised_eta', 'consignee_code_multiple', 'final_carrier_name']
    # Add transport_mode if it was used in filtering
    if modes and 'transport_mode' in results.columns:
        additional_cols.append('transport_mode')

    for c in additional_cols:
        if c in results.columns and c not in output_cols:
            output_cols.append(c)

    # Filter to available columns
    output_cols = [c for c in output_cols if c in results.columns]
    out = results[output_cols].head(200).copy()

    # ========== 11) FORMAT DATES ==========
    date_cols_to_format = ['etd_lp', 'atd_lp', 'eta_dp', 'revised_eta']
    for dcol in date_cols_to_format:
        if dcol in out.columns and pd.api.types.is_datetime64_any_dtype(out[dcol]):
            out[dcol] = out[dcol].dt.strftime('%Y-%m-%d')

    # ========== 12) RETURN RESULTS ==========
    try:
        logger.info(f"[get_containers_departing_from_load_port] Returning {len(out)} records")
    except:
        pass

    return out.where(pd.notnull(out), None).to_dict(orient='records')


def get_containers_still_at_load_port(question: str = None, consignee_code: str = None, **kwargs):
    """
    List containers that are STILL at a given load/origin port and have NOT yet departed.

    Core business logic:
      - atd_lp is NULL  => not departed from load port yet
      - load_port matches the user-provided location (name or code)

    Supports (query-driven) filters:
      - consignee_code (parameter) in addition to thread-local consignee filtering via _df()
      - hot containers: if query mentions hot/priority/urgent/rush/expedited
      - transport_mode: "by sea/air/road/rail/courier/sea-air"
      - optional ETD window ONLY if query explicitly mentions ETD/scheduled + a time phrase
        (e.g., "still at Shanghai with ETD next 7 days")

    Returns:
      - list[dict] of up to 200 rows (or a single count dict for "how many" queries)
    """

    query = (question or kwargs.get("query") or kwargs.get("input") or "").strip()
    if not query:
        return "Please provide a query like: 'Which containers are still at CHATTOGRAM and not yet departed?'"

    q_lower = query.lower()
    q_up = query.upper()

    # 1) Load data (already thread-local consignee scoped by _df())
    df = _df()
    if df is None or getattr(df, "empty", True):
        return "No data available for your authorized consignees."

    # 2) Optional explicit consignee_code narrowing (in addition to thread-local)
    if consignee_code and "consignee_code_multiple" in df.columns:
        codes = [c.strip().upper() for c in str(consignee_code).split(",") if c.strip()]
        code_set = set(codes) | {c.lstrip("0") for c in codes}

        def row_has_code(cell) -> bool:
            if pd.isna(cell):
                return False
            s = str(cell).upper()
            return any(c in s for c in code_set)

        df = df[df["consignee_code_multiple"].apply(row_has_code)].copy()
        if df.empty:
            return f"No containers found for consignee code(s) {', '.join(codes)}."

    # 3) Validate required columns
    if "load_port" not in df.columns:
        return "Load port column (load_port) not found in the dataset."
    if "atd_lp" not in df.columns:
        return "ATD load port column (atd_lp) not found in the dataset."

    # 4) Parse dates we may format/use
    date_cols = ["atd_lp"]
    if "etd_lp" in df.columns:
        date_cols.append("etd_lp")
    df = ensure_datetime(df, date_cols)

    # 5) Base filter: NOT YET DEPARTED from load port
    still_mask = df["atd_lp"].isna()
    df = df[still_mask].copy()
    if df.empty:
        return "No containers are currently pending departure (atd_lp is null) for your authorized consignees."

    # 6) Extract load port location (code or name) from query
    def normalize_port_name(port_str: str) -> str:
        if pd.isna(port_str):
            return ""
        s = str(port_str).upper()
        s = re.sub(r"\s*\([^)]*\)\s*", "", s)  # remove "(CNSHA)" part
        s = s.replace(",", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def extract_port_code(port_str: str) -> str | None:
        if pd.isna(port_str):
            return None
        m = re.search(r"\(([A-Z0-9\-]{3,6})\)", str(port_str).upper())
        return m.group(1) if m else None

    # Create normalized helpers for matching
    df["_port_name_norm"] = df["load_port"].apply(normalize_port_name)
    df["_port_code"] = df["load_port"].apply(extract_port_code)

    # (A) Explicit code like "(BDCGP)"
    port_code = None
    m_code = re.search(r"\(([A-Z0-9\-]{3,6})\)", q_up)
    if m_code:
        port_code = m_code.group(1).strip().upper()

    # (B) Bare code token like "BDCGP" (only accept if it exists in dataset codes)
    if not port_code:
        cand_codes = set(re.findall(r"\b[A-Z0-9\-]{3,6}\b", q_up))
        known_codes = set(df["_port_code"].dropna().astype(str).str.upper().unique().tolist())
        for c in sorted(cand_codes, key=len, reverse=True):
            if c in known_codes:
                port_code = c
                break

    # (C) Name phrase after "at/in/from/origin/load port"
    port_name = None
    if not port_code:
        patterns = [
            r"\b(?:still\s+at|at|in)\s+(?:load\s+port\s+|origin\s+port\s+|origin\s+)?([A-Za-z0-9\s\.\-\,\(\)]+?)(?=\s+\b(?:and|but|not|yet|depart|departed|leave|left|sail|sailed|ship|shipped|etd|today|tomorrow|yesterday|next|last|this|within|in)\b|[?.!,]|$)",
            r"\b(?:not\s+(?:yet\s+)?)?(?:departed|left|sailed)\s+from\s+([A-Za-z0-9\s\.\-\,\(\)]+?)(?=\s+\b(?:and|but|today|tomorrow|yesterday|next|last|this|within|in)\b|[?.!,]|$)",
            r"\bload\s+port\s+([A-Za-z0-9\s\.\-\,\(\)]+?)(?=\s+\b(?:and|but|today|tomorrow|yesterday|next|last|this|within|in)\b|[?.!,]|$)",
            r"\borigin\s+([A-Za-z0-9\s\.\-\,\(\)]+?)(?=\s+\b(?:and|but|today|tomorrow|yesterday|next|last|this|within|in)\b|[?.!,]|$)",
        ]
        for pat in patterns:
            m = re.search(pat, query, re.IGNORECASE)
            if m:
                port_name = m.group(1).strip()
                break

        if port_name:
            # Remove common trailing time words if captured
            port_name = re.sub(
                r"\b(?:today|tomorrow|yesterday|next|last|this|week|month|days?)\b.*$",
                "",
                port_name,
                flags=re.IGNORECASE,
            ).strip()

    if not port_code and not port_name:
        df.drop(columns=["_port_name_norm", "_port_code"], inplace=True, errors="ignore")
        return "Please specify a load port/location (e.g., 'at CHATTOGRAM' or '(BDCGP)')."

    # 7) Apply port filter (code > exact name > word-based > fuzzy)
    port_mask = pd.Series(False, index=df.index)

    if port_code:
        code_mask = df["_port_code"].fillna("").str.upper() == port_code
        port_mask |= code_mask
        try:
            logger.info(f"[get_containers_still_at_load_port] Code match '{port_code}': {int(code_mask.sum())} rows")
        except Exception:
            pass

    if not port_mask.any() and port_name:
        user_norm = normalize_port_name(port_name)
        exact = df["_port_name_norm"] == user_norm
        port_mask |= exact
        try:
            logger.info(f"[get_containers_still_at_load_port] Exact name match '{user_norm}': {int(exact.sum())} rows")
        except Exception:
            pass

        if not port_mask.any():
            words = [w for w in user_norm.split() if len(w) >= 3]
            if words:
                wmask = pd.Series(True, index=df.index)
                for w in words:
                    wmask &= df["_port_name_norm"].str.contains(rf"\b{re.escape(w)}\b", na=False, regex=True)
                port_mask |= wmask
                try:
                    logger.info(f"[get_containers_still_at_load_port] Word-based match {words}: {int(wmask.sum())} rows")
                except Exception:
                    pass
            else:
                sub = df["_port_name_norm"].str.contains(re.escape(user_norm), na=False)
                port_mask |= sub

        if not port_mask.any():
            # Fuzzy fallback over normalized unique names
            candidates = df["_port_name_norm"].dropna().astype(str).unique().tolist()
            best = get_close_matches(normalize_port_name(port_name), candidates, n=1, cutoff=0.85)
            if best:
                fuzzy = df["_port_name_norm"] == best[0]
                port_mask |= fuzzy
                try:
                    logger.info(f"[get_containers_still_at_load_port] Fuzzy match '{best[0]}': {int(fuzzy.sum())} rows")
                except Exception:
                    pass

    df = df[port_mask].copy()
    df.drop(columns=["_port_name_norm", "_port_code"], inplace=True, errors="ignore")

    if df.empty:
        where = port_code or port_name
        return f"No containers found still at load port '{where}' (atd_lp is null) for your authorized consignees."

    # 8) Optional transport mode filter (query-driven)
    try:
        modes = extract_transport_modes(query)
    except Exception:
        modes = set()

    if modes and "transport_mode" in df.columns:
        mode_mask = df["transport_mode"].astype(str).str.lower().apply(lambda s: any(m in s for m in modes))
        df = df[mode_mask].copy()
        if df.empty:
            where = port_code or port_name
            return f"No containers found still at '{where}' by transport mode {', '.join(sorted(modes))}."

    # 9) Optional hot filter (query-driven)
    is_hot_query = bool(re.search(r"\b(hot|priority|urgent|rush|expedited)\b", q_lower))
    hot_col = None
    if is_hot_query:
        hot_flag_cols = [c for c in df.columns if "hot_container_flag" in c.lower()] or \
                        [c for c in df.columns if c.lower() in ("hot_container", "hot_flag")]
        if hot_flag_cols:
            hot_col = hot_flag_cols[0]

            def _is_hot(v) -> bool:
                if pd.isna(v):
                    return False
                return str(v).strip().upper() in {"Y", "YES", "TRUE", "1", "HOT"}

            df = df[df[hot_col].apply(_is_hot)].copy()
            if df.empty:
                where = port_code or port_name
                return f"No hot containers found still at '{where}' (atd_lp is null)."

    # 10) Optional ETD window ONLY when query explicitly asks ETD/scheduled + time phrase
    wants_etd_window = ("etd" in q_lower) or ("scheduled" in q_lower) or ("schedule" in q_lower)
    mentions_time = bool(re.search(r"\b(today|tomorrow|yesterday|next|last|this|within|in)\b", q_lower))
    if wants_etd_window and mentions_time and "etd_lp" in df.columns:
        try:
            start_date, end_date, period_desc = parse_time_period(query)
            etd_norm = df["etd_lp"].dt.normalize()
            df = df[df["etd_lp"].notna() & (etd_norm >= start_date) & (etd_norm <= end_date)].copy()
            if df.empty:
                where = port_code or port_name
                return f"No not-yet-departed containers found at '{where}' with ETD in {period_desc}."
        except Exception:
            pass

    # 11) Output shaping
    output_cols = [
        "container_number",
        "load_port",
        "etd_lp",
        "atd_lp",
        "discharge_port",
        "po_number_multiple",
        "final_carrier_name",
        "transport_mode",
        "consignee_code_multiple",
    ]
    if hot_col and hot_col in df.columns:
        output_cols.append(hot_col)

    output_cols = [c for c in output_cols if c in df.columns]
    out = df[output_cols].drop_duplicates().copy()

    # Sort by ETD if present (otherwise by container)
    if "etd_lp" in out.columns:
        out = safe_sort_dataframe(out, "etd_lp", ascending=True)
    else:
        out = safe_sort_dataframe(out, "container_number", ascending=True)

    # Format dates
    for dcol in ["etd_lp", "atd_lp"]:
        if dcol in out.columns and pd.api.types.is_datetime64_any_dtype(out[dcol]):
            out[dcol] = out[dcol].dt.strftime("%Y-%m-%d")

    # 12) Count intent support
    is_count_query = bool(re.search(r"\b(how\s+many|count|number\s+of|total)\b", q_lower))
    if is_count_query:
        where = port_code or port_name
        sample = out["container_number"].dropna().astype(str).unique().tolist()[:50] if "container_number" in out.columns else []
        return [{
            "count": int(len(out)),
            "load_port": where,
            "logic": "atd_lp is null",
            "sample_containers": sample,
        }]

    return out.head(200).where(pd.notnull(out), None).to_dict(orient="records")


def get_containers_departed_from_load_port(query: str) -> str:
    """
    Get containers/POs/OBLs that have departed from specific load ports within a time period.

    Supports:
    - Container queries: "containers from SHANGHAI in last 7 days"
    - PO queries: "POs that departed from QINGDAO yesterday"
    - OBL queries: "BLs that left NINGBO last week"
    - Mixed queries: "show shipments from load port BUSAN in past 5 days"
    - Transport mode filtering: "containers departed by sea from HONG KONG"
    - Hot container filtering: "hot containers that left SHANGHAI last week"
    - Show me shipments that departed yesterday
    Uses ATD_LP when available; falls back to ETD_LP when ATD_LP is missing.
    Returns list[dict] with container/PO/OBL info and departure details.
    """

    query = (query or "").strip()
    query_upper = query.upper()
    query_lower = query.lower()

    # ========== 1) EXTRACT IDENTIFIERS (Container/PO/OBL) ==========
    # **CRITICAL FIX**: Remove consignee code mentions before extraction to avoid false positives
    query_for_extraction = query
    # Remove phrases like "for consignee 0028664", "consignee code 0028664", "consignee_code: 0028664", "user 0028664"
    query_for_extraction = re.sub(
        r'\b(?:for\s+)?(?:consignee|user)(?:\s*[_\-]?\s*code)?\s*[:=]?\s*\d{7}\b',
        '',
        query_for_extraction,
        flags=re.IGNORECASE,
    )

    container_no = extract_container_number(query_for_extraction)
    po_no = extract_po_number(query_for_extraction)
    obl_no = None
    try:
        obl_no = extract_ocean_bl_number(query_for_extraction)
    except Exception:
        pass

    try:
        logger.info(f"[get_containers_departed_from_load_port] Extracted: container={container_no}, po={po_no}, obl={obl_no}")
    except:
        pass

    # ========== 2) EXTRACT LOAD PORT ==========
    load_port = None

    # Pattern 1: "from load port PORTNAME" or "from PORTNAME"
    match = re.search(
        r'from\s+(?:load\s+port\s+)?([A-Za-z0-9\s,\-\(\)]+?)(?=\s+in\s+|\s+last\s+|\s+during\s+|\s+for\s+|\s+by\s+|[\?\.\,]|$)',
        query, re.IGNORECASE)
    if match:
        cand = match.group(1).strip()
        # Exclude common noise words
        if cand and cand.upper() not in ['CONSIGNEE', 'IN', 'LAST', 'THE', 'DAY', 'DAYS', 'NEXT', 'THIS', 'TODAY', 'TOMORROW', 'YESTERDAY', 'WEEK', 'MONTH', 'YEAR', 'SEA', 'AIR', 'ROAD']:
            load_port = cand.upper()

    # Pattern 2: "load port PORTNAME"
    if not load_port:
        match = re.search(
            r'load\s+port\s+([A-Za-z0-9\s,\-\(\)]+?)(?=\s+in\s+|\s+last\s+|\s+during\s+|\s+for\s+|\s+by\s+|[\?\.\,]|$)',
            query, re.IGNORECASE)
        if match:
            load_port = match.group(1).strip().upper()

    # Pattern 3: "departed/left from PORTNAME"
    if not load_port:
        match = re.search(
            r'(?:departed|left)\s+from\s+([A-Za-z0-9\s,\-\(\)]+?)(?=\s+in\s+|\s+last\s+|\s+during\s+|\s+for\s+|\s+by\s+|[\?\.\,]|$)',
            query, re.IGNORECASE)
        if match:
            cand = match.group(1).strip()
            if cand and cand.upper() not in ['CONSIGNEE', 'IN', 'LAST', 'THE', 'DAY', 'DAYS', 'NEXT', 'THIS', 'TODAY', 'TOMORROW', 'YESTERDAY', 'WEEK', 'MONTH', 'YEAR', 'SEA', 'AIR', 'ROAD']:
                load_port = cand.upper()

    # Pattern 4: Port with code in parentheses like "SHANGHAI(CNSHA)"
    if not load_port:
        match = re.search(r'([A-Za-z0-9\s,\-]+?\([A-Z0-9\-]{3,6}\))', query, re.IGNORECASE)
        if match:
            load_port = re.sub(r'\s+', ' ', match.group(1).strip()).upper()

    # Clean up extracted port name
    if load_port:
        load_port = re.sub(r'\s+', ' ', load_port).strip()

    try:
        logger.info(f"[get_containers_departed_from_load_port] Extracted load_port: '{load_port}'")
    except:
        pass

    # ========== 3) PARSE TIME WINDOW ==========
    start_date, end_date, period_desc = parse_time_period(query)

    try:
        logger.info(f"[get_containers_departed_from_load_port] Time window: {start_date} to {end_date} ({period_desc})")
    except:
        pass

    # ========== 4) LOAD AND FILTER DATA ==========
    df = _df()  # Respects consignee filtering

    try:
        if hasattr(threading.current_thread(), 'consignee_codes'):
            codes = threading.current_thread().consignee_codes
            logger.info(f"[get_containers_departed_from_load_port] Authorized consignee codes: {codes}, rows after filtering: {len(df)}")
        else:
            logger.info(f"[get_containers_departed_from_load_port] No consignee filtering (returning all {len(df)} rows)")
    except:
        pass

    if df.empty:
        return "No data available for your authorized consignees."

    # ========== 5) APPLY IDENTIFIER FILTERS (Container/PO/OBL) ==========
    identifier_mask = pd.Series(True, index=df.index)
    identifier_type = None

    # Filter by Container (highest priority)
    if container_no:
        if 'container_number' not in df.columns:
            return f"Container column not found for container {container_no}."

        clean_cont = clean_container_number(container_no)
        cont_col_norm = df["container_number"].astype(str).str.replace(r'[^A-Z0-9]', '', regex=True)
        identifier_mask = cont_col_norm == clean_cont

        if not identifier_mask.any():
            identifier_mask = df["container_number"].astype(str).str.contains(container_no, case=False, na=False)

        identifier_type = "container"
        try:
            logger.info(f"[get_containers_departed_from_load_port] Container filter: {identifier_mask.sum()} rows matched")
        except:
            pass

    # Filter by PO
    elif po_no:
        po_col = "po_number_multiple" if "po_number_multiple" in df.columns else ("po_number" if "po_number" in df.columns else None)
        if not po_col:
            return f"PO column not found for PO {po_no}."

        po_norm = _normalize_po_token(po_no)
        identifier_mask = df[po_col].apply(lambda cell: _po_in_cell(cell, po_norm))
        identifier_type = "PO"

        try:
            logger.info(f"[get_containers_departed_from_load_port] PO filter: {identifier_mask.sum()} rows matched")
        except:
            pass

    # Filter by OBL
    elif obl_no:
        bl_col = _find_ocean_bl_col(df)
        if not bl_col:
            return f"Ocean BL column not found for OBL {obl_no}."

        bl_norm = _normalize_bl_token(obl_no)
        identifier_mask = df[bl_col].apply(lambda cell: _bl_in_cell(cell, bl_norm))
        identifier_type = "OBL"

        try:
            logger.info(f"[get_containers_departed_from_load_port] OBL filter: {identifier_mask.sum()} rows matched")
        except:
            pass

    # Apply identifier filter
    df = df[identifier_mask].copy()

    if df.empty:
        if identifier_type:
            return f"No data found for {identifier_type} {container_no or po_no or obl_no}."

    # ========== NEW: 5A) TRANSPORT MODE FILTER ==========
    modes = extract_transport_modes(query)
    if modes:
        try:
            logger.info(f"[get_containers_departed_from_load_port] Transport modes detected: {modes}")
        except:
            pass

        if 'transport_mode' in df.columns:
            mode_mask = df['transport_mode'].astype(str).str.lower().apply(lambda s: any(m in s for m in modes))
            df = df[mode_mask].copy()

            try:
                logger.info(f"[get_containers_departed_from_load_port] After transport mode filter: {len(df)} rows")
            except:
                pass

            if df.empty:
                mode_str = ", ".join(sorted(modes))
                desc = f"{identifier_type} {container_no or po_no or obl_no}" if identifier_type else "containers"
                return f"No {desc} departed by {mode_str}."
        else:
            try:
                logger.warning(f"[get_containers_departed_from_load_port] transport_mode column not found, skipping mode filter")
            except:
                pass

    # ========== NEW: 5B) HOT CONTAINER FLAG FILTER ==========
    is_hot_query = bool(re.search(r'\bhot\b', query, re.IGNORECASE))
    if is_hot_query:
        try:
            logger.info(f"[get_containers_departed_from_load_port] Hot container filter requested")
        except:
            pass

        hot_flag_cols = [c for c in df.columns if 'hot_container_flag' in c.lower()]
        if not hot_flag_cols:
            hot_flag_cols = [c for c in df.columns if 'hot_container' in c.lower()]

        if hot_flag_cols:
            hot_col = hot_flag_cols[0]

            def _is_hot(v):
                if pd.isna(v):
                    return False
                return str(v).strip().upper() in {"Y", "YES", "TRUE", "1", "HOT"}

            hot_mask = df[hot_col].apply(_is_hot)
            df = df[hot_mask].copy()

            try:
                logger.info(f"[get_containers_departed_from_load_port] After hot flag filter: {len(df)} rows")
            except:
                pass

            if df.empty:
                desc = f"{identifier_type} {container_no or po_no or obl_no}" if identifier_type else "containers"
                return f"No hot {desc} found."
        else:
            try:
                logger.warning(f"[get_containers_departed_from_load_port] hot_container_flag column not found")
            except:
                pass

    # ========== 6) APPLY LOAD PORT FILTER (IMPROVED) ==========
    if load_port:
        if 'load_port' not in df.columns:
            return "Load port column not found in the dataset."

        # **CRITICAL FIX**: More flexible port matching
        def normalize_port_name(port_str):
            """Extract just the port name, removing code in parentheses"""
            if pd.isna(port_str):
                return ""
            s = str(port_str).upper()
            # Remove code in parentheses (e.g., "HONG KONG(HKHKG)" -> "HONG KONG")
            s = re.sub(r'\s*\([^)]*\)\s*', '', s)
            s = s.replace(',', ' ')
            s = re.sub(r'\s+', ' ', s).strip()
            return s

        def extract_port_code(port_str):
            """Extract port code from parentheses"""
            if pd.isna(port_str):
                return None
            m = re.search(r'\(([A-Z0-9\-]{3,6})\)', str(port_str).upper())
            return m.group(1) if m else None

        # Normalize user input
        user_input_norm = normalize_port_name(load_port)

        # Check if user provided a port code (3-6 alphanumeric)
        user_code = None
        m_code = re.search(r'\(([A-Z0-9\-]{3,6})\)', load_port.upper())
        if m_code:
            user_code = m_code.group(1)
        elif re.fullmatch(r'[A-Z0-9\-]{3,6}', user_input_norm.replace(' ', '')):
            user_code = user_input_norm.replace(' ', '')

        # Create temporary columns for matching
        df['_port_name_norm'] = df['load_port'].apply(normalize_port_name)
        df['_port_code'] = df['load_port'].apply(extract_port_code)

        try:
            logger.info(f"[get_containers_departed_from_load_port] Port matching: user_input='{load_port}', "
                       f"normalized='{user_input_norm}', code='{user_code}'")
        except:
            pass

        # **MATCHING STRATEGIES** (order matters - most specific first)
        port_mask = pd.Series(False, index=df.index)

        # Strategy 1: Exact port code match (highest priority)
        if user_code:
            code_mask = df['_port_code'].fillna('').str.upper() == user_code
            port_mask |= code_mask
            try:
                logger.info(f"[get_containers_departed_from_load_port] Code match '{user_code}': {code_mask.sum()} rows")
            except:
                pass

        # Strategy 2: Exact normalized name match
        exact_name_mask = df['_port_name_norm'] == user_input_norm
        port_mask |= exact_name_mask
        try:
            logger.info(f"[get_containers_departed_from_load_port] Exact name match '{user_input_norm}': {exact_name_mask.sum()} rows")
        except:
            pass

        # Strategy 3: Substring match (each word must be present)
        if not port_mask.any():
            words = [w for w in user_input_norm.split() if len(w) >= 2]  # Reduced from 3 to 2 for better matching
            if words:
                word_mask = pd.Series(True, index=df.index)
                for w in words:
                    word_mask &= df['_port_name_norm'].str.contains(w, na=False, regex=False)
                port_mask |= word_mask
                try:
                    logger.info(f"[get_containers_departed_from_load_port] Word-based match {words}: {word_mask.sum()} rows")
                except:
                    pass

        # Strategy 4: Fuzzy match on original load_port column (fallback)
        if not port_mask.any():
            fuzzy_mask = df['load_port'].astype(str).str.upper().str.contains(user_input_norm, na=False, regex=False)
            port_mask |= fuzzy_mask
            try:
                logger.info(f"[get_containers_departed_from_load_port] Fuzzy match: {fuzzy_mask.sum()} rows")
            except:
                pass

        # Apply port filter
        df = df[port_mask].copy()

        # Clean up temporary columns
        df.drop(columns=['_port_name_norm', '_port_code'], inplace=True, errors='ignore')

        try:
            logger.info(f"[get_containers_departed_from_load_port] After port filter: {len(df)} rows matched")
            if len(df) > 0:
                sample_ports = df['load_port'].head(5).tolist()
                logger.info(f"[get_containers_departed_from_load_port] Sample matched ports: {sample_ports}")
        except:
            pass

        if df.empty:
            desc = f"{identifier_type} {container_no or po_no or obl_no}" if identifier_type else "containers"
            return f"No {desc} found from load port {load_port}."

    # ========== 7) VALIDATE AND PARSE DATE COLUMNS ==========
    needed_cols = [c for c in ['atd_lp'] if c in df.columns]
    if not needed_cols:
        return "No departure date columns (atd_lp) found."

    df = ensure_datetime(df, needed_cols)

    # ========== 8) CREATE DEPARTURE DATE (ATD else ETD) ==========
    df['dep_for_filter'] = None
    if 'atd_lp' in df.columns:
        df['dep_for_filter'] = df['atd_lp'].where(df['atd_lp'].notna())


    # Mark source (ATD vs ETD)
    df['departure_source'] = df.apply(
        lambda r: 'ATD' if pd.notna(r.get('atd_lp')) else None, axis=1)


    # ========== 9) FILTER BY TIME WINDOW ==========
    try:
        logger.info(f"[get_containers_departed_from_load_port] Date filtering: start={start_date}, end={end_date}, period={period_desc}")
        logger.info(f"[get_containers_departed_from_load_port] Before date filter: {len(df)} rows")
        if len(df) > 0:
            # Show sample departure dates before filtering
            sample_dates = df[['container_number', 'dep_for_filter', 'departure_source']].head(10)
            logger.info(f"[get_containers_departed_from_load_port] Sample departure dates before filter:\n{sample_dates}")
    except:
        pass

    date_mask = (
        df['dep_for_filter'].notna()
        & (df['dep_for_filter'].dt.normalize() >= start_date)
        & (df['dep_for_filter'].dt.normalize() <= end_date)
    )
    results = df[date_mask].copy()

    try:
        logger.info(f"[get_containers_departed_from_load_port] After date filter: {len(results)} rows matched")
        if len(results) > 0:
            sample_filtered = results[['container_number', 'dep_for_filter', 'departure_source']].head(10)
            logger.info(f"[get_containers_departed_from_load_port] Sample filtered results:\n{sample_filtered}")
    except:
        pass

    if results.empty:
        desc = f"{identifier_type} {container_no or po_no or obl_no}" if identifier_type else "containers"
        port_desc = f" from load port {load_port}" if load_port else ""
        mode_desc = f" by {', '.join(sorted(modes))}" if modes else ""
        hot_desc = " (hot)" if is_hot_query else ""
        return f"No {desc}{hot_desc} departed{port_desc}{mode_desc} in the {period_desc}."

    # ========== 10) SORT BY DEPARTURE DATE (MOST RECENT FIRST) ==========
    results = results.sort_values('dep_for_filter', ascending=False)

    # ========== 11) PREPARE OUTPUT COLUMNS ==========
    output_cols = ['container_number', 'load_port', 'atd_lp', 'dep_for_filter', 'departure_source']

    # Add PO/OBL columns if relevant
    if identifier_type == "PO" or not identifier_type:
        po_col = "po_number_multiple" if "po_number_multiple" in results.columns else ("po_number" if "po_number" in results.columns else None)
        if po_col:
            output_cols.append(po_col)

    if identifier_type == "OBL" or not identifier_type:
        bl_col = _find_ocean_bl_col(results)
        if bl_col:
            output_cols.append(bl_col)

    # Add additional context columns
    additional_cols = ['discharge_port', 'revised_eta', 'consignee_code_multiple', 'final_carrier_name']

    # Add transport_mode if it was used in filtering
    if modes and 'transport_mode' in results.columns:
        additional_cols.append('transport_mode')

    # Add hot_container_flag if it was used in filtering
    if is_hot_query:
        hot_cols = [c for c in results.columns if 'hot_container_flag' in c.lower() or 'hot_container' in c.lower()]
        if hot_cols:
            additional_cols.append(hot_cols[0])

    for c in additional_cols:
        if c in results.columns and c not in output_cols:
            output_cols.append(c)

    # Filter to available columns
    output_cols = [c for c in output_cols if c in results.columns]
    out = results[output_cols].head(200).copy()

    # ========== 12) FORMAT DATES ==========
    date_cols_to_format = ['atd_lp', 'dep_for_filter', 'revised_eta']
    for dcol in date_cols_to_format:
        if dcol in out.columns and pd.api.types.is_datetime64_any_dtype(out[dcol]):
            out[dcol] = out[dcol].dt.strftime('%Y-%m-%d')

    # ========== 13) RENAME FOR CLARITY ==========
    out = out.rename(columns={'dep_for_filter': 'departure_date'})

    # ========== 14) RETURN RESULTS ==========
    try:
        logger.info(f"[get_containers_departed_from_load_port] Returning {len(out)} records")
    except:
        pass

    return out.where(pd.notnull(out), None).to_dict(orient='records')


def get_containers_missed_planned_etd(query: str) -> str:
    """Get containers that missed their planned ETD from load port.

    Business logic (as requested):
    - atd_lp (actual departure from load port) is NULL
    - etd_lp (estimated departure from load port) is < sysdate (today)

    Notes:
    - Respects consignee filtering via _df() (thread-local consignee codes)
    - Returns list[dict] (up to 200 records)
    """

    _ = (query or "").strip()

    df = _df()
    if df.empty:
        return "No data available for your authorized consignees."

    if "etd_lp" not in df.columns:
        return "ETD load port column (etd_lp) not found."
    if "atd_lp" not in df.columns:
        return "ATD load port column (atd_lp) not found."

    df = ensure_datetime(df, ["etd_lp", "atd_lp"])
    today = pd.Timestamp.today().normalize()

    mask = (
        df["etd_lp"].notna()
        & df["atd_lp"].isna()
        & (df["etd_lp"].dt.normalize() < today)
    )
    results = df[mask].copy()

    if results.empty:
        return "No containers have missed their planned ETD for your authorized consignees."

    results = results.sort_values("etd_lp", ascending=True)

    output_cols = [
        "container_number",
        "load_port",
        "discharge_port",
        "etd_lp",
        "atd_lp",
    ]
    for col in [
        "consignee_code_multiple",
        "po_number_multiple",
        "final_carrier_name",
        "transport_mode",
        "hot_container_flag",
    ]:
        if col in results.columns and col not in output_cols:
            output_cols.append(col)

    output_cols = [c for c in output_cols if c in results.columns]
    out = results[output_cols].head(200).copy()

    for dcol in ["etd_lp", "atd_lp"]:
        if dcol in out.columns and pd.api.types.is_datetime64_any_dtype(out[dcol]):
            out[dcol] = out[dcol].dt.strftime("%Y-%m-%d")

    return out.where(pd.notnull(out), None).to_dict(orient="records")


def get_container_carrier(input_str: str) -> str:
    """
    Get the carrier information for a specific container or PO.
    Input: Query should mention a container number or PO number (partial or full).
    Output: Carrier details including carrier name, code, and SCAC code if available.
    For PO queries: Shows final carrier. If multiple Show latest w.r.t. ETD/ETA.
    If no container/PO is found, prompts for a valid identifier.
    """
    # Try to extract container number first
    container_no = extract_container_number(input_str)

    # Try to extract PO number if no container found
    po_no = None
    if not container_no:
        po_no = extract_po_number(input_str)

    # --- NEW: if user asked only about a PO (no container), delegate to PO-specific function ---
    if po_no and not container_no:
        return get_carrier_for_po(input_str)

    if not container_no and not po_no:
        return "Please specify a valid container number or PO number to get carrier information."

    df = _df()  # This automatically filters by consignee

    # Search by container number first
    if container_no:
        df["container_number"] = df["container_number"].astype(str)

        # Exact match after normalizing
        clean = clean_container_number(container_no)
        rows = df[df["container_number"].str.replace(" ", "").str.upper() == clean]

        # Fallback to contains match
        if rows.empty:
            rows = df[df["container_number"].str.contains(container_no, case=False, na=False)]

        identifier = f"container {container_no}"

        if rows.empty:
            return f"No data found for {identifier} or you are not authorized to access this container."

        # For containers, use the first match
        row = rows.iloc[0]

    # Search by PO number if container not found
    elif po_no:
        po_col = "po_number_multiple" if "po_number_multiple" in df.columns else "po_number"
        if po_col in df.columns:
            # Search for PO in the comma-separated field
            rows = df[df[po_col].astype(str).str.contains(po_no, case=False, na=False)]
            identifier = f"PO {po_no}"
        else:
            return "PO number column not found in the data."

        if rows.empty:
            return f"No data found for {identifier} or you are not authorized to access this PO."

        # For PO queries: If multiple records, show latest w.r.t. ETD/ETA
        if len(rows) > 1:
            # Ensure datetime columns are properly formatted
            rows = ensure_datetime(rows, ["etd_lp", "etd_flp", "eta_dp", "eta_fd"])

            # Create a combined date field for sorting (latest ETD/ETA)
            date_cols = ["etd_lp", "etd_flp", "eta_dp", "eta_fd"]
            available_date_cols = [col for col in date_cols if col in rows.columns]

            if available_date_cols:
                # Get the latest date from available date columns for each row
                latest_dates = []
                for idx, row_data in rows.iterrows():
                    row_dates = [row_data[col] for col in available_date_cols if pd.notnull(row_data[col])]
                    if row_dates:
                        latest_dates.append((idx, max(row_dates)))
                    else:
                        latest_dates.append((idx, pd.Timestamp.min))

                # Sort by latest date and get the most recent
                if latest_dates:
                    latest_dates.sort(key=lambda x: x[1], reverse=True)
                    latest_idx = latest_dates[0][0]
                    row = rows.loc[latest_idx]
                    selection_info = f" (latest from {len(rows)} records based on ETD/ETA)"
                else:
                    row = rows.iloc[0]
                    selection_info = f" (first from {len(rows)} records)"
            else:
                row = rows.iloc[0]
                selection_info = f" (first from {len(rows)} records)"
        else:
            row = rows.iloc[0]
            selection_info = ""

    # Build carrier information response
    carrier_info = []

    # Primary carrier information from final_carrier_name
    if (
        "final_carrier_name" in row.index
        and pd.notnull(row["final_carrier_name"])
        and str(row["final_carrier_name"]).strip()
    ):
        carrier_info.append(f"Final Carrier Name: {row['final_carrier_name']}")

    # Additional carrier details
    carrier_fields = [
        ("final_carrier_code", "Final Carrier Code"),
        ("final_carrier_scac_code", "Final Carrier SCAC Code"),
        ("true_carrier_code", "True Carrier Code"),
        ("true_carrier_scac_code", "True Carrier SCAC Code"),
    ]

    for field, label in carrier_fields:
        if field in row.index and pd.notnull(row[field]) and str(row[field]).strip():
            carrier_info.append(f"{label}: {row[field]}")

    if not carrier_info:
        return f"No carrier information available for {identifier}."

    # Build response
    response_lines = []
    if po_no and len(rows) > 1:
        response_lines.append(f"Carrier information for {identifier}{selection_info}:")
    else:
        response_lines.append(f"Carrier information for {identifier}:")

    response_lines.extend(carrier_info)

    # Add additional context for PO queries with selection details
    if po_no and 'selection_info' in locals() and len(rows) > 1:
        # Show the date used for selection if available
        date_info = []
        for col in ["etd_lp", "etd_flp", "eta_dp", "eta_fd"]:
            if col in row.index and pd.notnull(row[col]):
                date_val = row[col]
                if hasattr(date_val, "strftime"):
                    date_str = date_val.strftime("%Y-%m-%d")
                else:
                    date_str = str(date_val)
                col_name = col.replace("_", " ").upper()
                date_info.append(f"{col_name}: {date_str}")

        if date_info:
            response_lines.append("")
            response_lines.append("Selection based on latest date from:")
            response_lines.append(f"- {', '.join(date_info)}")

    return "\n".join(response_lines)


def get_containers_by_etd_window(question: str = None, consignee_code: str = None, **kwargs) -> str:
    """
    List containers whose ETD (etd_lp) falls within the requested time window.
    - Uses parse_time_period(question) for the time window (today, next 7 days, ranges, etc.)
    - Uses etd_lp only (ETD). Does not exclude rows with ATD.
    - Optional filters inferred from the query:
        * load_port (port name or code in parentheses, e.g. SHANGHAI or (CNSHA))
        * hot containers (if 'hot' is mentioned)
        * transport mode (sea/air/road/rail/courier via extract_transport_modes)
    - Optional consignee filtering via parameter (consignee_code)
    Returns list[dict] with container_number, load_port, etd_lp, and context.
    """

    query = (question or "").strip()
    q_up = query.upper()

    try:
        logger.info(f"[get_containers_by_etd_window] Processing query: {query}")
    except:
        pass

    # 1) Parse time window
    start_date, end_date, period_desc = parse_time_period(query)

    try:
        logger.info(f"[get_containers_by_etd_window] Parsed time window: {start_date} to {end_date} ({period_desc})")
    except:
        pass

    # 2) Load dataset (thread-local consignee restriction already applied)
    df = _df()
    if df.empty:
        return "No container records available."

    try:
        logger.info(f"[get_containers_by_etd_window] Initial dataset size: {len(df)} rows")
    except:
        pass

    # 3) Optional explicit consignee filter (codes like 0000866)
    if consignee_code and "consignee_code_multiple" in df.columns:
        codes = [c.strip().upper() for c in str(consignee_code).split(",") if c.strip()]
        code_set = set(codes) | {c.lstrip("0") for c in codes}

        def row_has_code(cell):
            if pd.isna(cell):
                return False
            s = str(cell).upper()
            if any(re.search(rf"\({re.escape(c)}\)", s) for c in code_set):
                return True
            return any(re.search(rf"\b{re.escape(c)}\b", s) for c in code_set)

        df = df[df["consignee_code_multiple"].apply(row_has_code)].copy()
        if df.empty:
            return f"No containers found for consignee code(s) {', '.join(codes)}."

        try:
            logger.info(f"[get_containers_by_etd_window] After consignee filter: {len(df)} rows")
        except:
            pass

    # 4) Validate columns and parse dates
    if "etd_lp" not in df.columns:
        return "No ETD column (etd_lp) found to compute the ETD window."

    date_cols = ["etd_lp"]
    if "atd_lp" in df.columns:
        date_cols.append("atd_lp")

    df = ensure_datetime(df, date_cols)

    # 8) load_port (origin) filtering from query (code or name)
    # **CRITICAL FIX**: Improved port extraction to avoid capturing too much text
    lp_col = "load_port" if "load_port" in df.columns else None
    if lp_col:
        port_code = None
        name_phrase = None

        # Pattern 1: Code in parentheses like (CNSHA)
        m_code = re.search(r"\(([A-Z0-9]{3,6})\)", q_up)
        if m_code:
            port_code = m_code.group(1).strip().upper()

        # Pattern 2: Port code after "FROM" - FIXED to capture only port code/name
        if not port_code:
            # Match "FROM CNSHA" or "FROM SHANGHAI" - stop at common boundary words
            m_name = re.search(
                r"\bFROM\s+(?:LOAD\s+PORT\s+)?([A-Z][A-Z0-9\s\.\-]{2,15}?)(?=\s+(?:FOR|IN|ON|AT|WITH|TO|ETD|TODAY|TOMORROW|NEXT|WITHIN|YESTERDAY|LAST|THIS|CONSIGNEE|\d+)|[\?\.\,]|\s*$)",
                q_up
            )
            if m_name:
                cand = m_name.group(1).strip()
                # Clean up any trailing noise
                cand = re.sub(r"(?:ETD|TODAY|TOMORROW|NEXT|WITHIN|FOR|CONSIGNEE).*$", "", cand).strip()
                if len(cand) >= 3:
                    name_phrase = cand

        # Pattern 3: "AT|IN|TO PORT_NAME" - similar fix
        if not port_code and not name_phrase:
            m_name = re.search(
                r"\b(?:AT|IN|TO)\s+(?:LOAD\s+PORT\s+)?([A-Z][A-Z0-9\s\.\-]{2,15}?)(?=\s+(?:FOR|FROM|WITH|ETD|TODAY|TOMORROW|NEXT|WITHIN|YESTERDAY|LAST|THIS|CONSIGNEE|\d+)|[\?\.\,]|\s*$)",
                q_up
            )
            if m_name:
                cand = m_name.group(1).strip()
                cand = re.sub(r"(?:ETD|TODAY|TOMORROW|NEXT|WITHIN|FOR|CONSIGNEE).*$", "", cand).strip()
                if len(cand) >= 3:
                    name_phrase = cand

        # Pattern 4: Bare port code (3-6 uppercase letters) - validate against known codes
        if not port_code and not name_phrase:
            # Extract known port codes from dataset
            known_codes = set()
            try:
                sample_ports = df[lp_col].dropna().astype(str).head(1000)
                for port_str in sample_ports:
                    codes_in_port = re.findall(r'\(([A-Z0-9]{3,6})\)', str(port_str).upper())
                    known_codes.update(codes_in_port)
            except:
                pass

            # Look for standalone port codes in query
            candidate_codes = re.findall(r'\b([A-Z]{3,6})\b', q_up)
            skip_words = {"ETD", "TODAY", "TOMORROW", "NEXT", "DAYS", "FROM", "LOAD", "PORT", "WITH", "FOR", "CONSIGNEE"}

            for code in candidate_codes:
                if code not in skip_words and (not known_codes or code in known_codes):
                    port_code = code
                    break

        try:
            logger.info(f"[get_containers_by_etd_window] Port extraction: code={port_code}, name={name_phrase}")
        except:
            pass

        if port_code or name_phrase:
            def _norm_port(s):
                if pd.isna(s):
                    return ""
                t = str(s).upper()
                t = re.sub(r"\([^)]*\)", "", t)
                t = t.replace(",", " ")
                return re.sub(r"\s+", " ", t).strip()

            lp_series = df[lp_col].astype(str)

            if port_code:
                # Try matching port code in parentheses first
                lp_mask = lp_series.str.upper().str.contains(rf"\({re.escape(port_code)}\)", na=False)

                # If no match, try matching bare code
                if not lp_mask.any():
                    lp_mask = lp_series.str.upper().str.contains(rf"\b{re.escape(port_code)}\b", na=False)
            else:
                lp_norm = lp_series.apply(_norm_port)
                phrase_norm = re.sub(r"\s+", " ", name_phrase or "").strip()

                # Try exact match first
                exact = (lp_norm == phrase_norm)
                if exact.any():
                    lp_mask = exact
                else:
                    # Try word-by-word matching
                    words = [w for w in phrase_norm.split() if len(w) >= 3]
                    if words:
                        cond = pd.Series(True, index=df.index)
                        for w in words:
                            cond &= lp_norm.str.contains(re.escape(w), na=False)
                        lp_mask = cond
                    else:
                        # Fallback to substring match
                        lp_mask = lp_norm.str.contains(re.escape(phrase_norm), na=False)

                # Final fallback: raw substring match on original column
                if not lp_mask.any():
                    lp_mask = lp_series.str.upper().str.contains(re.escape(phrase_norm), na=False)

            df = df[lp_mask].copy()

            try:
                logger.info(f"[get_containers_by_etd_window] After port filter: {len(df)} rows matched")
                if len(df) > 0:
                    sample_ports = df[lp_col].head(5).tolist()
                    logger.info(f"[get_containers_by_etd_window] Sample matching ports: {sample_ports}")
            except:
                pass

            if df.empty:
                return f"No containers found from port {port_code or name_phrase}."

    # 5) Base date mask: ETD inside window
    etd_norm = df["etd_lp"].dt.normalize()
    mask = df["etd_lp"].notna() & (etd_norm >= start_date) & (etd_norm <= end_date)

    # 6) Hot containers filter
    if re.search(r"\bHOT\b", q_up):
        hot_cols = [c for c in df.columns if 'hot_container_flag' in c.lower()] or \
                   [c for c in df.columns if 'hot_container' in c.lower()]
        if not hot_cols:
            return "Hot container flag column not found in the data."
        hot_col = hot_cols[0]

        def _is_hot(v):
            if pd.isna(v):
                return False
            s = str(v).strip().upper()
            return s in {"Y", "YES", "TRUE", "1", "HOT"} or v is True or v == 1

        mask &= df[hot_col].apply(_is_hot)

    # 7) Transport mode filter
    try:
        modes = extract_transport_modes(query)
    except Exception:
        modes = set()
    if modes and "transport_mode" in df.columns:
        mode_mask = df["transport_mode"].astype(str).str.lower().apply(lambda s: any(m in s for m in modes))
        mask &= mode_mask

    # 9) Apply mask
    result = df[mask].copy()

    if result.empty:
        return f"No containers with ETD between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}."

    # 10) Prepare output
    out_cols = [
        "container_number",
        "load_port",
        "etd_lp",
        "atd_lp",
        "discharge_port",
        "po_number_multiple",
        "final_carrier_name",
        "consignee_code_multiple",
    ]
    if "transport_mode" in result.columns:
        out_cols.append("transport_mode")
    hot_cols = [c for c in result.columns if c.lower() in ("hot_container_flag", "hot_container")]
    if hot_cols:
        out_cols.append(hot_cols[0])

    out_cols = [c for c in out_cols if c in result.columns]
    out = result[out_cols].sort_values("etd_lp", ascending=True).head(200).copy()

    # Format dates
    for dcol in ["etd_lp", "atd_lp"]:
        if dcol in out.columns and pd.api.types.is_datetime64_any_dtype(out[dcol]):
            out[dcol] = out[dcol].dt.strftime("%Y-%m-%d")

    return out.where(pd.notnull(out), None).to_dict(orient="records")


def get_upcoming_bls(question: str = None, consignee_code: str = None, **kwargs) -> str:
    """
    List upcoming ocean BLs (ocean_bl_no_multiple) with ETA filtering.
    - Extracts numeric/alphanumeric BL tokens (comma-separated, e.g., MAEU255513671, 61110045884, JKT25-00327/BSN).
    - Supports delay queries ("delayed by X days", "late by more than Y days").
    - Supports hot BL filtering via hot_container_flag column.
    - Supports transport mode filtering (sea, air, etc.).
    - Supports location filtering (port code or city name).
    - Respects consignee_code (comma-separated) and consignee name detection.
    Returns list[dict] with columns: ocean_bl_no_multiple, container_number, discharge_port, revised_eta, eta_dp, delay_days (if delayed), hot_container_flag, transport_mode.
    """

    query = (question or "").strip()
    query_upper = query.upper()

    # **FIX**: Use parse_time_period() and extract start_date, end_date
    start_date, end_date, period_desc = parse_time_period(query)

    try:
        logger.info(f"[get_upcoming_bls] Period: {period_desc}, "
                   f"Dates: {format_date_for_display(start_date)} to "
                   f"{format_date_for_display(end_date)}")
    except:
        pass

    df = _df()

    # Find BL column robustly
    bl_col = _find_ocean_bl_col(df)
    if not bl_col:
        return "Ocean BL column (ocean_bl_no_multiple) not found in dataset."

    # Apply consignee_code filter (supports comma-separated codes)
    if consignee_code and "consignee_code_multiple" in df.columns:
        cc = str(consignee_code).strip().upper()
        cc_list = [c.strip().upper() for c in cc.split(',') if c.strip()]
        cc_set = set(cc_list)

        def row_has_code(cell):
            if pd.isna(cell):
                return False
            parts = {p.strip().upper() for p in re.split(r',\s*', str(cell)) if p.strip()}
            for part in parts:
                if part in cc_set:
                    return True
                m = re.search(r'\(([A-Z0-9\- ]+)\)\s*$', part)
                if m:
                    code = m.group(1).strip().upper()
                    if code in cc_set or code.lstrip('0') in cc_set or code in {c.lstrip('0') for c in cc_set}:
                        return True
            return False

        df = df[df['consignee_code_multiple'].apply(row_has_code)].copy()
        if df.empty:
            return f"No BL records found for consignee code(s) {', '.join(cc_list)}."

    # Detect consignee name in query
    if 'consignee_code_multiple' in df.columns:
        try:
            all_cons_parts = set()
            token_to_name = {}
            token_to_code = {}

            for raw in df['consignee_code_multiple'].dropna().astype(str).tolist():
                for part in re.split(r',\s*', raw):
                    p = part.strip()
                    if not p:
                        continue
                    tok = p.upper()
                    all_cons_parts.add(tok)
                    m_code = re.search(r'\(([A-Z0-9\- ]+)\)\s*$', tok)
                    code = m_code.group(1).strip() if m_code else None
                    name_part = re.sub(r'\([^\)]*\)', '', tok).strip()
                    token_to_name[tok] = name_part
                    token_to_code[tok] = code

            sorted_cons = sorted(all_cons_parts, key=lambda x: -len(x))
            mentioned_consignee_tokens = set()

            for cand in sorted_cons:
                if re.search(r'\b' + re.escape(cand) + r'\b', query_upper):
                    mentioned_consignee_tokens.add(cand)

            numeric_tokens = [t for t in sorted_cons if token_to_code.get(t) and re.fullmatch(r'\d+', token_to_code[t])]
            if numeric_tokens:
                num_q_tokens = re.findall(r'\b0*\d+\b', query_upper)
                for qnum in num_q_tokens:
                    for tok in numeric_tokens:
                        code = token_to_code.get(tok)
                        if not code:
                            continue
                        if qnum == code or qnum.lstrip('0') == code.lstrip('0'):
                            mentioned_consignee_tokens.add(tok)

            q_clean = re.sub(r'[^A-Z0-9\s]', ' ', query_upper)
            q_words = [w for w in re.split(r'\s+', q_clean) if len(w) >= 3]

            for tok in sorted_cons:
                name_part = token_to_name.get(tok, "")
                if not name_part:
                    continue
                if re.search(r'\b' + re.escape(name_part) + r'\b', query_upper):
                    mentioned_consignee_tokens.add(tok)
                    continue
                matches = sum(1 for w in q_words if w in name_part)
                if matches >= 2 or any((w in name_part and len(w) >= 4) for w in q_words):
                    mentioned_consignee_tokens.add(tok)

            m_for = re.search(r'\bFOR\s+([A-Z0-9\&\.\-\s]{3,})', query_upper)
            if m_for:
                target = m_for.group(1).strip()
                for tok in sorted_cons:
                    name_part = token_to_name.get(tok, "")
                    if target in name_part or target == tok:
                        mentioned_consignee_tokens.add(tok)

            if mentioned_consignee_tokens:
                strict_keys = set()
                for tok in mentioned_consignee_tokens:
                    strict_keys.add(tok)
                    code = token_to_code.get(tok)
                    name = token_to_name.get(tok)
                    if code:
                        strict_keys.add(code)
                        strict_keys.add(code.lstrip('0'))
                    if name:
                        strict_keys.add(name)
                strict_keys = {k.upper().strip() for k in strict_keys if k}

                def row_has_cons_strict(cell):
                    if pd.isna(cell):
                        return False
                    parts = [p.strip().upper() for p in re.split(r',\s*', str(cell)) if p.strip()]
                    for p in parts:
                        if p in strict_keys:
                            return True
                        m = re.search(r'\(([A-Z0-9\- ]+)\)\s*$', p)
                        if m:
                            code_in_cell = m.group(1).strip().upper()
                            if code_in_cell in strict_keys or code_in_cell.lstrip('0') in strict_keys:
                                return True
                        name_part = re.sub(r'\([^\)]*\)', '', p).strip().upper()
                        if name_part and name_part in strict_keys:
                            return True
                    return False

                df = df[df['consignee_code_multiple'].apply(row_has_cons_strict)].copy()
                if df.empty:
                    return f"No BL records for consignee {', '.join(sorted(mentioned_consignee_tokens))}."

        except Exception:
            pass

    # Hot container filtering
    is_hot_query = bool(re.search(r'\bhot\b', query, re.IGNORECASE))
    hot_flag_cols = [c for c in df.columns if 'hot_container_flag' in c.lower()]
    if not hot_flag_cols:
        hot_flag_cols = [c for c in df.columns if 'hot_container' in c.lower()]

    if is_hot_query:
        if not hot_flag_cols:
            return "Hot container flag column not found in the data."
        hot_col = hot_flag_cols[0]

        def _is_hot(v):
            if pd.isna(v):
                return False
            s = str(v).strip().upper()
            return s in {"Y", "YES", "TRUE", "1", "HOT"} or v is True or v == 1

        df = df[df[hot_col].apply(_is_hot)].copy()
        if df.empty:
            return "No hot BLs found for your authorized consignees."

    # Transport mode filtering
    modes = extract_transport_modes(query)
    if modes and 'transport_mode' in df.columns:
        df = df[df['transport_mode'].astype(str).str.lower().apply(lambda s: any(m in s for m in modes))].copy()
        if df.empty:
            return f"No BLs found for transport mode(s): {', '.join(sorted(modes))}."

    # Location filtering
    port_cols = [c for c in ["discharge_port", "vehicle_arrival_lcn", "final_destination", "place_of_delivery"] if c in df.columns]
    if port_cols:
        location_mask = pd.Series(False, index=df.index, dtype=bool)
        location_found = False
        location_name = None

        candidate_tokens = re.findall(r'\b[A-Z0-9]{3,6}\b', query_upper)
        skip_tokens = {"NEXT", "DAYS", "IN", "AT", "ON", "THE", "AND", "TO", "FROM", "ARRIVE", "ARRIVING", "HOT", "OBL", "BL"}
        candidate_tokens = [t for t in candidate_tokens if t not in skip_tokens and not t.isdigit()]

        def row_contains_code(port_string, token):
            if pd.isna(port_string):
                return False
            s = str(port_string).upper()
            if re.search(r'\(' + re.escape(token) + r'\)', s):
                return True
            if re.search(r'\b' + re.escape(token) + r'\b', s):
                return True
            extracted = re.findall(r'\(([A-Z0-9]{3,6})\)', s)
            if extracted and token in extracted:
                return True
            return False

        for tok in candidate_tokens:
            tok_mask = pd.Series(False, index=df.index)
            for col in port_cols:
                tok_mask |= df[col].apply(lambda s, t=tok: row_contains_code(s, t))
            if tok_mask.any():
                location_mask = tok_mask
                location_found = True
                location_name = tok
                break

        if not location_found:
            paren = re.search(r'\(([A-Z0-9]{3,6})\)', query_upper)
            if paren:
                tok = paren.group(1)
                tok_mask = pd.Series(False, index=df.index)
                for col in port_cols:
                    tok_mask |= df[col].apply(lambda s, t=tok: row_contains_code(s, t))
                if tok_mask.any():
                    location_mask = tok_mask
                    location_found = True
                    location_name = tok

        if not location_found:
            city_patterns = [
                r'(?:at|in|to)\s+([A-Za-z\s\.\-]{3,})(?:[,\s]|$)',
                r'\b(LOS ANGELES|LONG BEACH|SINGAPORE|ROTTERDAM|HONG KONG|SHANGHAI|BUSAN|TOKYO|OAKLAND|SAVANNAH|NLRTM)\b'
            ]
            for patt in city_patterns:
                m = re.search(patt, query, re.IGNORECASE)
                if m:
                    city = m.group(1).strip().upper()
                    location_name = city
                    city_mask = pd.Series(False, index=df.index)
                    for col in port_cols:
                        def match_city(port_string, city=city):
                            if pd.isna(port_string):
                                return False
                            s = str(port_string).upper()
                            cleaned = re.sub(r'\([^)]*\)', '', s).strip()
                            return city in cleaned
                        city_mask |= df[col].apply(match_city)
                    if city_mask.any():
                        location_mask = city_mask
                        location_found = True
                        break

        if location_found:
            df = df[location_mask].copy()
            if df.empty:
                return f"No BLs found at {location_name}."

    # DATE selection and filtering
    date_priority = [c for c in ['revised_eta', 'eta_dp'] if c in df.columns]
    if not date_priority:
        return "No ETA columns (revised_eta / eta_dp) found in the data."

    parse_cols = date_priority.copy()
    if 'ata_dp' in df.columns:
        parse_cols.append('ata_dp')
    df = ensure_datetime(df, parse_cols)

    # Delay detection and filtering
    is_delay_query = any(w in query.lower() for w in ("delay", "late", "overdue", "behind", "missed"))

    if is_delay_query:
        # Filter for arrived containers (ata_dp not null)
        arrived = df[df['ata_dp'].notna()].copy()
        if arrived.empty:
            return "No BLs have arrived for your authorized consignees."

        arrived["delay_days"] = (arrived["ata_dp"] - arrived["eta_dp"]).dt.days.fillna(0).astype(int)

        range_match = re.search(r"(\d{1,4})\s*[-–—]\s*(\d{1,4})\s*days?\b", query, re.IGNORECASE)
        less_than = re.search(r"\b(?:less\s+than|under|below|<)\s*(\d{1,4})\s*days?\b", query, re.IGNORECASE)
        more_than = re.search(r"\b(?:more\s+than|over|>)\s*(\d{1,4})\s*days?\b", query, re.IGNORECASE)
        plus_sign = re.search(r"\b(\d{1,4})\s*\+\s*days?\b", query, re.IGNORECASE)
        exact = re.search(r"\b(?:delayed|late|overdue|behind)\s+by\s+(\d{1,4})\s+days?\b", query, re.IGNORECASE)

        if range_match:
            d1, d2 = int(range_match.group(1)), int(range_match.group(2))
            low, high = min(d1, d2), max(d1, d2)
            delayed = arrived[(arrived["delay_days"] >= low) & (arrived["delay_days"] <= high)]
        elif less_than:
            d = int(less_than.group(1))
            delayed = arrived[(arrived["delay_days"] > 0) & (arrived["delay_days"] < d)]
        elif more_than or plus_sign:
            d = int((more_than or plus_sign).group(1))
            delayed = arrived[arrived["delay_days"] > d]
        elif exact:
            d = int(exact.group(1))
            delayed = arrived[arrived["delay_days"] == d]
        else:
            delayed = arrived[arrived["delay_days"] > 0]

        if delayed.empty:
            return "No delayed BLs found for your authorized consignees."

        # Group by BL and aggregate
        out_cols = [bl_col, "container_number", "eta_dp", "ata_dp", "delay_days", "discharge_port", "consignee_code_multiple"]
        if hot_flag_cols:
            out_cols.append(hot_flag_cols[0])
        if 'transport_mode' in delayed.columns:
            out_cols.append('transport_mode')
        out_cols = [c for c in out_cols if c in delayed.columns]

        agg_dict = {
            "container_number": lambda s: ", ".join(sorted(set(s.dropna().astype(str)))),
            "delay_days": "max",
            "eta_dp": "first",
            "ata_dp": "first",
            "discharge_port": "first",
            "consignee_code_multiple": "first"
        }
        if hot_flag_cols and hot_flag_cols[0] in delayed.columns:
            agg_dict[hot_flag_cols[0]] = "first"
        if 'transport_mode' in delayed.columns:
            agg_dict['transport_mode'] = "first"

        result = delayed.groupby(bl_col).agg(agg_dict).reset_index()

        for dcol in ["eta_dp", "ata_dp"]:
            if dcol in result.columns and pd.api.types.is_datetime64_any_dtype(result[dcol]):
                result[dcol] = result[dcol].dt.strftime("%Y-%m-%d")

        return result.where(pd.notnull(result), None).to_dict(orient="records")

    # Upcoming arrivals (not yet arrived)
    if 'revised_eta' in df.columns and 'eta_dp' in df.columns:
        df['eta_for_filter'] = df['revised_eta'].where(df['revised_eta'].notna(), df['eta_dp'])
    elif 'revised_eta' in df.columns:
        df['eta_for_filter'] = df['revised_eta']
    else:
        df['eta_for_filter'] = df['eta_dp']

    # **FIX**: Use start_date and end_date from parse_time_period()
    today = pd.Timestamp.today().normalize()

    date_mask = df['eta_for_filter'].notna() & (df['eta_for_filter'] >= start_date) & (df['eta_for_filter'] <= end_date)
    if 'ata_dp' in df.columns:
        date_mask &= df['ata_dp'].isna()

    result = df[date_mask].copy()
    if result.empty:
        loc_str = f" at {location_name}" if location_found else ""
        return f"No BLs arriving{loc_str} between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}."

    # Group by BL and aggregate
    out_cols = [bl_col, "container_number", "discharge_port", "revised_eta", "eta_dp", "eta_for_filter", "consignee_code_multiple"]
    if hot_flag_cols:
        out_cols.append(hot_flag_cols[0])
    if 'transport_mode' in result.columns:
        out_cols.append('transport_mode')
    out_cols = [c for c in out_cols if c in result.columns]

    agg_dict = {
        "container_number": lambda s: ", ".join(sorted(set(s.dropna().astype(str)))),
        "discharge_port": "first",
        "revised_eta": "first",
        "eta_dp": "first",
        "eta_for_filter": "first",
        "consignee_code_multiple": "first"
    }
    if hot_flag_cols and hot_flag_cols[0] in result.columns:
        agg_dict[hot_flag_cols[0]] = "first"
    if 'transport_mode' in result.columns:
        agg_dict['transport_mode'] = "first"

    final_result = result.groupby(bl_col).agg(agg_dict).reset_index().sort_values('eta_for_filter').head(200)

    for d in ['revised_eta', 'eta_dp', 'eta_for_filter']:
        if d in final_result.columns and pd.api.types.is_datetime64_any_dtype(final_result[d]):
            final_result[d] = final_result[d].dt.strftime('%Y-%m-%d')

    if 'eta_for_filter' in final_result.columns:
        final_result = final_result.drop(columns=['eta_for_filter'])

    return final_result.where(pd.notnull(final_result), None).to_dict(orient='records')


def vector_search_tool(query: str) -> str:
    """
    Search the vector database for relevant shipment information using semantic similarity.
    Input: Natural language query.
    Output: Top matching records from the vector store.
    """
    # Get the vectorstore instance
    vectorstore = get_vectorstore()
    # Search for top 5 relevant documents
    results = vectorstore.similarity_search(query, k=5)
    if not results:
        return "No relevant results found in the vector database."
    # Format results for display
    details = "\n\n".join([str(doc.page_content) for doc in results])
    return (
        f"Thought: Retrieved top results from the vector database using semantic search.\n"
        f"Final Answer:\n{details}\n"
        "These results are based on semantic similarity from the vector store."
    )

def get_blob_sql_engine():
    """
    Load the shipment CSV into a persistent SQLite DB and return an engine.
    Avoid circular imports by building the engine here.
    """

    df = get_shipment_df().copy()
    engine = create_engine("sqlite:///shipment_blob.db", echo=False, future=True)
    with engine.begin() as conn:
        df.to_sql("shipment", conn, if_exists="replace", index=False)
    return engine

# def get_blob_sql_engine():
#     """
#     Loads the shipment CSV from Azure Blob and creates a persistent SQLite engine for SQL queries.
#     """
#     return get_persistent_sql_engine()


def get_sql_agent():
    """Get the SQL agent instance with proper error handling."""
    from agents.azure_agent import initialize_sql_agent
    return initialize_sql_agent()


def sql_query_tool(natural_language_query: str) -> str:
    """
    Execute natural language queries against the shipment database.

    Examples:
    - "Show me containers delayed by more than 3 days"
    - "How many containers are arriving this week?"
    - "Which carrier handles the most shipments?"
    - "List all containers from Singapore port"

    You don't need to write SQL - just ask in plain English!
    """
    try:

        logger.info(f"Processing natural language query: {natural_language_query}")
        sql_agent = initialize_sql_agent()

        if not sql_agent:
            return "Error: SQL agent is not available. Please check the database connection."

        # The agent will automatically convert natural language to SQL
        result = sql_agent.run(natural_language_query)
        logger.info("Natural language query processed successfully")
        return str(result)

    except Exception as exc:
        error_msg = f"Failed to process natural language query: {str(exc)}"
        logger.error(error_msg, exc_info=True)
        return f"Error: {error_msg}"


#-----------------new functions added for additional tools-----------------

def _normalize_po_token(s: str) -> str:
    """Normalize a PO token for comparison: strip, upper, keep alphanumerics."""
    if s is None:
        return ""
    s = str(s).strip().upper()
    # Keep alphanumeric only (common PO formats), remove surrounding/inline junk
    s = re.sub(r'[^A-Z0-9]', '', s)
    return s

def _po_in_cell(cell: str, po_norm: str) -> bool:
    """Return True if normalized PO exists in a comma/sep-separated cell."""
    if pd.isna(cell) or po_norm == "":
        return False
    # split on common separators
    parts = re.split(r'[,;/\|\s]+', str(cell))
    for p in parts:
        if _normalize_po_token(p) == po_norm:
            return True
    return False

def extract_transport_modes(query: str) -> set:
    """
    Parse transport mode tokens from a user query and return normalized set
    e.g. "sea", "air", "road", "rail", "courier", "sea-air".
    """
    if not query:
        return set()
    q = query.lower()
    mapping = {
        "sea-air": "sea-air",
        "sea air": "sea-air",
        "sea": "sea",
        "ocean": "sea",
        "air": "air",
        "airfreight": "air",
        "air freight": "air",
        "courier": "courier",
        "rail": "rail",
        "road": "road",
        "truck": "road",
        "trucking": "road",
        "multimodal": "sea-air",
        "intermodal": "sea-air"
    }
    found = set()
    for key, norm in mapping.items():
        if key in q:
            found.add(norm)
    return found


def get_containers_by_transport_mode(query: str) -> str:
    """
    Handle queries about transport_mode (e.g. "containers arrived by sea",
    "which container will arrive by air in next 3 days").
    Behaviour:
    - Detect transport mode(s) from query using extract_transport_modes().
    - If query contains a next/N-days window -> return upcoming arrivals filtered by transport_mode.
    - Otherwise treat as 'arrived' request and return rows with ata_dp not null filtered by transport_mode.
    - Uses _df() (consignee filtering), ensure_datetime, and per-row ETA logic where needed.
    """

    modes = extract_transport_modes(query)
    if not modes:
        return "No transport mode detected in the query."

    df = _df()
    if 'transport_mode' not in df.columns:
        return "No 'transport_mode' column in data."

    # filter by transport mode (case-insensitive substring match)
    df_mode = df[df['transport_mode'].astype(str).str.lower().apply(lambda s: any(m in s for m in modes))].copy()
    if df_mode.empty:
        return f"No containers found for transport mode(s): {', '.join(sorted(modes))} for your authorized consignees."

    # If user asked for upcoming window -> delegate to arriving-soon logic here (but operate on df_mode)
    m_days = re.search(r'(?:next|in|upcoming|within)\s+(\d{1,3})\s+days?', query, re.IGNORECASE)
    if m_days:
        days = int(m_days.group(1))
        # use per-row ETA preference
        date_priority = [c for c in ['revised_eta', 'eta_dp'] if c in df_mode.columns]
        if not date_priority:
            return "No ETA columns (revised_eta / eta_dp) found to compute upcoming arrivals."
        parse_cols = date_priority.copy()
        if 'ata_dp' in df_mode.columns:
            parse_cols.append('ata_dp')
        df_mode = ensure_datetime(df_mode, parse_cols)

        if 'revised_eta' in df_mode.columns and 'eta_dp' in df_mode.columns:
            df_mode['eta_for_filter'] = df_mode['revised_eta'].where(df_mode['revised_eta'].notna(), df_mode['eta_dp'])
        elif 'revised_eta' in df_mode.columns:
            df_mode['eta_for_filter'] = df_mode['revised_eta']
        else:
            df_mode['eta_for_filter'] = df_mode['eta_dp']

        today = pd.Timestamp.today().normalize()
        future = today + pd.Timedelta(days=days)
        mask = df_mode['eta_for_filter'].notna() & (df_mode['eta_for_filter'] >= today) & (df_mode['eta_for_filter'] <= future)
        if 'ata_dp' in df_mode.columns:
            mask &= df_mode['ata_dp'].isna()
        out = df_mode[mask].copy()
        if out.empty:
            return f"No containers by {', '.join(sorted(modes))} arriving between {today.strftime('%Y-%m-%d')} and {future.strftime('%Y-%m-%d')}."
        cols = [c for c in ['container_number', 'po_number_multiple', 'discharge_port', 'revised_eta', 'eta_dp', 'eta_for_filter'] if c in out.columns]
        out = out[cols].sort_values('eta_for_filter').head(50).copy()
        for d in ['revised_eta', 'eta_dp', 'eta_for_filter']:
            if d in out.columns and pd.api.types.is_datetime64_any_dtype(out[d]):
                out[d] = out[d].dt.strftime('%Y-%m-%d')
        if 'eta_for_filter' in out.columns:
            out = out.drop(columns=['eta_for_filter'])
        return out.where(pd.notnull(out), None).to_dict(orient='records')


    # Otherwise treat as "arrived by <mode>" -> return rows with ata_dp not null
    if 'ata_dp' not in df_mode.columns:
        return "No ATA column (ata_dp) present to determine arrived containers."
    df_mode = ensure_datetime(df_mode, ['ata_dp', 'revised_eta', 'eta_dp'])
    arrived = df_mode[df_mode['ata_dp'].notna()].copy()
    if arrived.empty:
        return f"No containers have arrived by {', '.join(sorted(modes))} for your authorized consignees."
    cols = [c for c in ['container_number', 'po_number_multiple', 'discharge_port', 'ata_dp', 'final_carrier_name'] if c in arrived.columns]
    arrived = arrived[cols].sort_values('ata_dp', ascending=False).head(100).copy()
    if 'ata_dp' in arrived.columns and pd.api.types.is_datetime64_any_dtype(arrived['ata_dp']):
        arrived['ata_dp'] = arrived['ata_dp'].dt.strftime('%Y-%m-%d')
    return arrived.where(pd.notnull(arrived), None).to_dict(orient='records')


def get_carrier_for_po(query: str) -> str:
    """
    Find final_carrier_name for a PO.
    Accepts queries like "who is carrier for PO 5500009022" or "5500009022" or alphanumeric POs (e.g. 7196461A).
    Looks up PO in po_number_multiple (comma-separated) or po_number and returns final_carrier_name and container.
    """
    # try helper extractor first, fallback to generic alnum token (6-12 chars)
    po = extract_po_number(query)
    if not po:
        m = re.search(r'\b([A-Z0-9]{6,12})\b', query.upper())
        po = m.group(1) if m else None
    if not po:
        return "Please specify a PO number"

    po_norm = _normalize_po_token(po)
    df = _df()
    po_col = "po_number_multiple" if "po_number_multiple" in df.columns else ("po_number" if "po_number" in df.columns else None)
    if not po_col:
        return "PO column not found in the dataset."

    # locate rows where any token in the PO column matches exactly (after normalization)
    mask = df[po_col].apply(lambda cell: _po_in_cell(cell, po_norm))
    matches = df[mask].copy()
    if matches.empty:
        return f"No data found for PO {po}."

    # if many matches, pick the most-relevant by latest date among common date columns
    date_priority = ["revised_eta", "eta_dp", "eta_fd", "predictive_eta", "etd_lp", "etd_flp"]
    available_date_cols = [c for c in date_priority if c in matches.columns]
    if available_date_cols:
        matches = ensure_datetime(matches, available_date_cols)
        # compute max date per row (NaT -> ignored)
        matches["_row_max_date"] = matches[available_date_cols].max(axis=1)
        # choose row with latest date (rows with all NaT get Timestamp.min)
        matches["_row_max_date"] = matches["_row_max_date"].fillna(pd.Timestamp.min)
        chosen = matches.sort_values("_row_max_date", ascending=False).iloc[0]
        matches = matches.drop(columns=["_row_max_date"], errors="ignore")
    else:
        chosen = matches.iloc[0]

    container = chosen.get("container_number", "<unknown>")
    carrier = None
    if "final_carrier_name" in chosen.index and pd.notnull(chosen["final_carrier_name"]):
        carrier = str(chosen["final_carrier_name"]).strip()

    if carrier:
        return f"The carrier for PO {po} (container {container}) is {carrier}."
    else:
        return f"Carrier (final_carrier_name) not found for PO {po} (container {container})."


def _normalize_po_token(s: str) -> str:
    """Normalize a PO token for comparison: strip, upper, keep alphanumerics."""
    if s is None:
        return ""
    s = str(s).strip().upper()
    # Keep alphanumeric only (common PO formats), remove surrounding/inline junk
    s = re.sub(r'[^A-Z0-9]', '', s)
    return s

def _po_in_cell(cell: str, po_norm: str) -> bool:
    """Return True if normalized PO exists in a comma/sep-separated cell."""
    if pd.isna(cell) or po_norm == "":
        return False
    # split on common separators
    parts = re.split(r'[,;/\|\s]+', str(cell))
    for p in parts:
        if _normalize_po_token(p) == po_norm:
            return True
    return False

def get_carrier_for_po(query: str) -> str:
    """
    Find final_carrier_name for a PO.
    Accepts queries like "who is carrier for PO 5500009022" or "5500009022" or alphanumeric POs (e.g. 7196461A).
    Looks up PO in po_number_multiple (comma-separated) or po_number and returns final_carrier_name and container.
    """
    # try helper extractor first, fallback to generic alnum token (6-12 chars)
    po = extract_po_number(query)
    if not po:
        m = re.search(r'\b([A-Z0-9]{6,12})\b', query.upper())
        po = m.group(1) if m else None
    if not po:
        return "Please specify a PO number."

    po_norm = _normalize_po_token(po)
    df = _df()
    po_col = "po_number_multiple" if "po_number_multiple" in df.columns else ("po_number" if "po_number" in df.columns else None)
    if not po_col:
        return "PO column not found in the dataset."

    # locate rows where any token in the PO column matches exactly (after normalization)
    mask = df[po_col].apply(lambda cell: _po_in_cell(cell, po_norm))
    matches = df[mask].copy()
    if matches.empty:
        return f"No data found for PO {po}."

    # if many matches, pick the most-relevant by latest date among common date columns
    date_priority = ["revised_eta", "eta_dp", "eta_fd", "predictive_eta", "etd_lp", "etd_flp"]
    available_date_cols = [c for c in date_priority if c in matches.columns]
    if available_date_cols:
        matches = ensure_datetime(matches, available_date_cols)
        # compute max date per row (NaT -> ignored)
        matches["_row_max_date"] = matches[available_date_cols].max(axis=1)
        # choose row with latest date (rows with all NaT get Timestamp.min)
        matches["_row_max_date"] = matches["_row_max_date"].fillna(pd.Timestamp.min)
        chosen = matches.sort_values("_row_max_date", ascending=False).iloc[0]
        matches = matches.drop(columns=["_row_max_date"], errors="ignore")
    else:
        chosen = matches.iloc[0]

    container = chosen.get("container_number", "<unknown>")
    carrier = None
    if "final_carrier_name" in chosen.index and pd.notnull(chosen["final_carrier_name"]):
        carrier = str(chosen["final_carrier_name"]).strip()

    if carrier:
        return f"The carrier for PO {po} (container {container}) is {carrier}."
    else:
        return f"Carrier (final_carrier_name) not found for PO {po} (container {container})."


def is_po_hot(query: str, as_records: bool = False) -> str:
    """
    Check whether a PO is marked hot via the container's hot flag.
    Returns a short sentence listing related containers and which are hot.
    Strictly treat as container-only if a real container number exists (AAAA#########).

    If as_records=True, returns list[dict] of matching rows with key columns instead of a sentence.
    """

    # 1) Only consider container path if a true container token is present
    m_con = re.search(r"\b([A-Z]{4}\d{7})\b", str(query).upper())
    if m_con:
        container_no = m_con.group(1)
        df = _df()
        if 'container_number' not in df.columns:
            return "Container number column not found in the dataset."
        norm_con = container_no.replace(" ", "").upper()
        try:
            con_mask = df['container_number'].astype(str).str.replace(" ", "", regex=False).str.upper() == norm_con
        except Exception:
            con_mask = df['container_number'].astype(str).str.upper() == norm_con
        matches = df[con_mask].copy()
        if matches.empty:
            return f"No data found for container {container_no}."
        # Identify hot-flag column
        hot_flag_cols = [c for c in df.columns if 'hot_container_flag' in c.lower()]
        if not hot_flag_cols:
            hot_flag_cols = [c for c in df.columns if 'hot_container' in c.lower()]
        if not hot_flag_cols:
            return "No hot-container flag column found in the dataset."
        hot_col = hot_flag_cols[0]
        def _is_hot_container(v):
            if pd.isna(v):
                return False
            return str(v).strip().upper() in {"Y", "YES", "TRUE", "1", "HOT"}
        matches = matches.assign(_is_hot=matches[hot_col].apply(_is_hot_container))

        # Ensure eta_dp is parsed for formatting
        if 'eta_dp' in matches.columns:
            matches = ensure_datetime(matches, ['eta_dp'])

        # Build detail records for hot rows
        det_cols = [c for c in ['container_number', 'po_number_multiple', 'discharge_port', 'eta_dp', hot_col, '_is_hot', 'consignee_code_multiple'] if c in matches.columns]
        hot_rows = matches[matches['_is_hot']].copy()
        details = hot_rows[det_cols].copy() if not hot_rows.empty else matches[det_cols].copy()

        # Ensure column presence and order (always include these keys)
        desired_cols = ['container_number', 'po_number_multiple', 'discharge_port', 'eta_dp', hot_col, '_is_hot', 'consignee_code_multiple']
        for c in desired_cols:
            if c not in details.columns:
                details[c] = None
        details = details[[c for c in desired_cols if c in details.columns]]

        if 'eta_dp' in details.columns and pd.api.types.is_datetime64_any_dtype(details['eta_dp']):
            details['eta_dp'] = details['eta_dp'].dt.strftime('%Y-%m-%d')

        # Optional dict output
        if as_records:
            return details.where(pd.notnull(details), None).to_dict(orient='records')

        return (
            f"Container {container_no} is HOT."
            f" Details: {details.where(pd.notnull(details), None).to_dict(orient='records')}"
            if matches["_is_hot"].any()
            else f"Container {container_no} is not marked hot."
        )

    # 2) PO path (default)
    from utils.container import extract_po_number
    po = extract_po_number(query)
    if not po:
        # fallback: capture a 6–12 length alphanumeric that is not a container
        m = re.search(r'\b([A-Z0-9]{6,12})\b', str(query).upper())
        po = m.group(1) if m else None
    if not po:
        return "Please specify a PO number."

    po_norm = _normalize_po_token(po)
    df = _df()
    po_col = "po_number_multiple" if "po_number_multiple" in df.columns else ("po_number" if "po_number" in df.columns else None)
    if not po_col:
        return "PO column not found in the dataset."

    # Robust multi-value PO matching
    mask = df[po_col].apply(lambda cell: _po_in_cell(cell, po_norm))
    matches = df[mask].copy()
    if matches.empty:
        return f"No data found for PO {po}."

    # Identify hot-flag column
    hot_flag_cols = [c for c in df.columns if 'hot_container_flag' in c.lower()]
    if not hot_flag_cols:
        hot_flag_cols = [c for c in df.columns if 'hot_container' in c.lower()]
    if not hot_flag_cols:
        return "No hot-container flag column found in the dataset."
    hot_col = hot_flag_cols[0]

    def _is_hot(v):
        if pd.isna(v):
            return False
        return str(v).strip().upper() in {"Y", "YES", "TRUE", "1", "HOT"}

    matches = matches.assign(_is_hot=matches[hot_col].apply(_is_hot))

    # Ensure eta_dp is parsed for formatting
    if 'eta_dp' in matches.columns:
        matches = ensure_datetime(matches, ['eta_dp'])

    # Detail records for hot subset
    det_cols = [c for c in [po_col, 'container_number', 'po_number_multiple', 'discharge_port', 'eta_dp', hot_col, '_is_hot', 'consignee_code_multiple'] if c in matches.columns]
    hot_subset = matches[matches['_is_hot']].copy()
    details = hot_subset[det_cols].copy() if not hot_subset.empty else matches[det_cols].copy()

    # Ensure column presence and order (always include these keys)
    desired_cols = ['container_number', 'po_number_multiple', 'discharge_port', 'eta_dp', hot_col, '_is_hot', 'consignee_code_multiple']
    for c in desired_cols:
        if c not in details.columns:
            details[c] = None
    details = details[[c for c in desired_cols if c in details.columns]]

    if 'eta_dp' in details.columns and pd.api.types.is_datetime64_any_dtype(details['eta_dp']):
        details['eta_dp'] = details['eta_dp'].dt.strftime('%Y-%m-%d')

    # Optional dict output
    if as_records:
        return details.where(pd.notnull(details), None).to_dict(orient='records')

    all_containers = sorted(matches["container_number"].dropna().astype(str).unique().tolist()) if "container_number" in matches.columns else []
    hot_containers = sorted(matches.loc[matches["_is_hot"], "container_number"].dropna().astype(str).unique().tolist()) if "container_number" in matches.columns else []

    if hot_containers:
        return (
            f"PO {po} is HOT on container(s): {', '.join(hot_containers)}. "
            f"Details: {details.where(pd.notnull(details), None).to_dict(orient='records')}."
            f" Related containers: {', '.join(all_containers)}."
        )
    else:
        related = f" Related containers: {', '.join(all_containers)}." if all_containers else ""
        return f"PO {po} is not marked hot.{related}"


def _normalize_bl_token(s: str) -> str:
    """Normalize an ocean BL token for comparison: uppercase alphanumerics only."""
    if s is None:
        return ""
    s = str(s).strip().upper()
    s = re.sub(r'[^A-Z0-9]', '', s)
    return s

def _bl_in_cell(cell: str, bl_norm: str) -> bool:
    """Return True if normalized BL exists in a comma/sep-separated cell."""
    if pd.isna(cell) or bl_norm == "":
        return False
    parts = re.split(r'[,;/\|\s]+', str(cell))
    for p in parts:
        if _normalize_bl_token(p) == bl_norm:
            return True
    return False

def _find_ocean_bl_col(df: pd.DataFrame) -> str | None:
    """
    Return the best-matching column name for the ocean BL field.
    Accepts variants like:
      - ocean_bl_no_multiple
      - Ocean BL No (Multiple)
      - Ocean BL No Multiple
      - ocean bl no multiple
    Minimal, robust matching: normalize column name (lower, remove non-alnum)
    and check for presence of key tokens.
    """
    for col in df.columns:
        key = re.sub(r'[^a-z0-9]', '', str(col).lower())
        # require at least 'ocean' and 'bl' and 'no' (or 'multiple') to be present
        if 'ocean' in key and 'bl' in key and ('no' in key or 'multiple' in key):
            return col
    # fallback: exact known name if present
    if 'ocean_bl_no_multiple' in df.columns:
        return 'ocean_bl_no_multiple'
    return None


def get_containers_for_bl(query: str) -> str:
    # ...existing docstring...
        # extract BL from query (prefer dedicated extractor, fallback to regex)
    bl = None
    try:
        bl = extract_ocean_bl_number(query)
    except Exception:
        bl = None
    if not bl:
        m = re.search(r'\b([A-Z0-9]{6,24})\b', query.upper())
        bl = m.group(1) if m else None
    if not bl:
        return "Please specify an ocean BL (e.g. 'MOLWMNL2400017')."

    bl_norm = _normalize_bl_token(bl)
    df = _df()

    # find BL column robustly (accepts "Ocean BL No (Multiple)" etc.)
    bl_col = _find_ocean_bl_col(df)
    if not bl_col:
        return "No ocean BL column (ocean_bl_no_multiple) found in dataset."
    norm_col = "_ocean_bl_norm"
    df[norm_col] = df[bl_col].astype(str).fillna("").str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)

    # match either via normalized column contains OR via existing tokenized helper (fallback)
    mask = df[norm_col].str.contains(bl_norm, na=False) | df[bl_col].apply(lambda cell: _bl_in_cell(cell, bl_norm))
    matches = df[mask].copy()
    # cleanup temporary column
    df.drop(columns=[norm_col], inplace=True, errors=True)
    if matches.empty:
        return f"No data found for ocean BL {bl}."
    # transport-mode filter (if present in query and dataset) — operate on matches for minimal impact
    modes = extract_transport_modes(query)
    if modes and 'transport_mode' in matches.columns:
        matches = matches[matches['transport_mode'].astype(str).str.lower().apply(lambda s: any(m in s for m in modes))].copy()
    if matches.empty:
         return f"No data found for ocean BL {bl}."

    # transport-mode filter (if present in query and dataset)
    # modes = extract_transport_modes(query)
    # if modes and 'transport_mode' in df.columns:
    #     df = df[df['transport_mode'].astype(str).str.lower().apply(lambda s: any(m in s for m in modes))]

    # bl_col = "ocean_bl_no_multiple" if "ocean_bl_no_multiple" in df.columns else None
    # if not bl_col:
    #     return "No ocean BL column (ocean_bl_no_multiple) found in dataset."

    # # find matching rows
    # mask = df[bl_col].apply(lambda cell: _bl_in_cell(cell, bl_norm))
    # matches = df[mask].copy()
    # if matches.empty:
    #     return f"No data found for ocean BL {bl}."
    # transport-mode filter (if present in query and dataset) — operate on matches for minimal impact
    modes = extract_transport_modes(query)
    if modes and 'transport_mode' in matches.columns:
        matches = matches[matches['transport_mode'].astype(str).str.lower().apply(lambda s: any(m in s for m in modes))].copy()
    if matches.empty:
        return f"No data found for ocean BL {bl}."

    # ensure relevant date columns
    date_cols = [c for c in ["ata_dp", "revised_eta", "eta_dp"] if c in matches.columns]
    if date_cols:
        matches = ensure_datetime(matches, date_cols)
        try:
            min_dt = matches[date_cols].min().min()
            max_dt = matches[date_cols].max().max()
            logger.info(f"[get_containers_for_bl] matched_rows={len(matches)} date_cols={date_cols} min_date={min_dt} max_date={max_dt}")
        except Exception:
            pass

    lines = []
    for _, row in matches.iterrows():
        cont = row.get("container_number", "<unknown>")
        dp = row.get("discharge_port", "<unknown discharge port>")
        ata = row.get("ata_dp", pd.NaT) if "ata_dp" in row.index else pd.NaT
        rev = row.get("revised_eta", pd.NaT) if "revised_eta" in row.index else pd.NaT
        eta = row.get("eta_dp", pd.NaT) if "eta_dp" in row.index else pd.NaT

        if pd.notna(ata):
            try:
                arrived_on = pd.to_datetime(ata).strftime("%Y-%m-%d")
            except Exception:
                arrived_on = str(ata)
            lines.append(f"Container {cont} (BL {bl}) arrived at {dp} on {arrived_on}.")
        else:
            # not arrived — show preferred ETA (revised_eta if present else eta_dp)
            preferred = None
            src = None
            if pd.notna(rev):
                try:
                    preferred = pd.to_datetime(rev).strftime("%Y-%m-%d")
                except Exception:
                    preferred = str(rev)
                src = "revised ETA"
            elif pd.notna(eta):
                try:
                    preferred = pd.to_datetime(eta).strftime("%Y-%m-%d")
                except Exception:
                    preferred = str(eta)
                src = "ETA"

            if preferred:
                lines.append(f"Container {cont} (BL {bl}) has {src} {preferred} for {dp} (not arrived yet).")
            else:
                lines.append(f"Container {cont} (BL {bl}) - no ETA/ATA available for {dp}.")

    return "\n".join(lines)

def get_carrier_for_bl(query: str) -> str:
    """
    Return the final_carrier_name for an ocean BL value.
    - Accepts 'MOLWMNL2400017' or 'who is carrier for BL MOLWMNL2400017'.
    - If multiple container rows match, picks the most-recent row using common date columns.
    """
    m = re.search(r'\b([A-Z0-9]{6,24})\b', query.upper())
    bl = m.group(1) if m else None
    if not bl:
        return "Please specify an ocean BL (e.g. 'MOLWMNL2400017')."

    bl_norm = _normalize_bl_token(bl)
    df = _df()
    bl_col = _find_ocean_bl_col(df)
    if not bl_col:
        return "No ocean BL column (ocean_bl_no_multiple) found in dataset."

    try:
        modes_dbg = sorted(list(extract_transport_modes(query)))
    except Exception:
        modes_dbg = []
    logger.info(f"[get_carrier_for_bl] query={query!r} bl={bl} parsed_modes={modes_dbg}")

    # transport-mode filter (if present)
    modes = extract_transport_modes(query)
    if modes and 'transport_mode' in df.columns:
        df = df[df['transport_mode'].astype(str).str.lower().apply(lambda s: any(m in s for m in modes))]

    # bl_col = "ocean_bl_no_multiple" if "ocean_bl_no_multiple" in df.columns else None
    # if not bl_col:
    #     return "No ocean BL column (ocean_bl_no_multiple) found in dataset."

    # mask = df[bl_col].apply(lambda cell: _bl_in_cell(cell, bl_norm))
    # locate matches using the robust finder / tokenizer
    mask = df[bl_col].apply(lambda cell: _bl_in_cell(cell, bl_norm))
    matches = df[mask].copy()
    if matches.empty:
        return f"No data found for ocean BL {bl}."

    # choose most relevant row by date priority
    date_priority = ["revised_eta", "eta_dp", "eta_fd", "predictive_eta", "etd_lp", "etd_flp"]
    avail = [c for c in date_priority if c in matches.columns]
    if avail:
        matches = ensure_datetime(matches, avail + (["ata_dp"] if "ata_dp" in matches.columns else []))
        try:
            logger.info(f"[get_carrier_for_bl] matched_rows={len(matches)} date_priority={avail}")
        except Exception:
           pass
        matches["_row_max"] = matches[avail].max(axis=1).fillna(pd.Timestamp.min)
        chosen = matches.sort_values("_row_max", ascending=False).iloc[0]
    else:
        chosen = matches.iloc[0]

    cont = chosen.get("container_number", "<unknown>")
    carrier = chosen.get("final_carrier_name") if "final_carrier_name" in chosen.index else None
    if pd.notna(carrier) and str(carrier).strip():
        return f"The carrier for BL {bl} (container {cont}) is {str(carrier).strip()}."
    else:
        return f"Carrier (final_carrier_name) not found for BL {bl} (container {cont})."

def is_bl_hot(query: str) -> str:
    """
    Check whether an ocean BL is marked hot (via its container's hot flag).
    Returns which related containers are HOT and lists all related containers.
    """
    # Try to extract BL from query (use dedicated extractor with fallback)
    bl = None
    try:
        bl = extract_ocean_bl_number(query)  # Use the dedicated extractor first
    except Exception:
        pass

    if not bl:  # Fallback to regex if extractor fails
        m = re.search(r'\b([A-Z0-9]{6,24})\b', query.upper())
        bl = m.group(1) if m else None

    if not bl:
        return "Please specify an ocean BL (e.g. 'MOLWMNL2400017')."

    bl_norm = _normalize_bl_token(bl)
    df = _df()  # This automatically filters by consignee

    # Find ocean BL column robustly (use helper if available, otherwise search for columns with 'bl')
    bl_col = None
    try:
        bl_col = _find_ocean_bl_col(df)
    except Exception:
        # Fallback - find columns containing 'ocean_bl', 'bl_no', etc.
        bl_candidates = [c for c in df.columns if any(x in c.lower() for x in ['ocean_bl', 'bl_no', 'bill_of_lading'])]
        bl_col = bl_candidates[0] if bl_candidates else "ocean_bl_no_multiple"

    if bl_col not in df.columns:
        return f"No ocean BL column ({bl_col}) found in dataset."

    # Create normalized column for matching
    norm_col = "_bl_norm"
    df[norm_col] = df[bl_col].astype(str).fillna("").str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)

    # Match using normalized column OR the cell tokenizer
    mask = df[norm_col].str.contains(bl_norm, na=False) | df[bl_col].apply(lambda cell: _bl_in_cell(cell, bl_norm))
    matches = df[mask].copy()
    df.drop(columns=[norm_col], inplace=True, errors="ignore")

    # Apply transport mode filter if specified
    modes = extract_transport_modes(query)
    if modes and 'transport_mode' in matches.columns:
        matches = matches[matches['transport_mode'].astype(str).str.lower().apply(lambda s: any(m in s for m in modes))].copy()

    if matches.empty:
        logger.info(f"[is_bl_hot] No matches for BL={bl} (normalized={bl_norm}) in {bl_col}")
        return f"No data found for ocean BL {bl}."

    # Find hot flag column(s)
    hot_cols = [c for c in df.columns if "hot_container_flag" in c.lower()]
    if not hot_cols:
        hot_cols = [c for c in df.columns if "hot_container" in c.lower()]

    if not hot_cols:
        return "No hot-container flag column found in the dataset."

    hot_col = hot_cols[0]

    def is_hot_val(v):
        if pd.isna(v):
            return False
        return str(v).strip().upper() in {"Y", "YES", "TRUE", "1", "HOT"}

    matches = matches.assign(_is_hot=matches[hot_col].apply(is_hot_val))
    all_containers = sorted(matches["container_number"].dropna().astype(str).unique().tolist())
    hot_containers = sorted(matches.loc[matches["_is_hot"], "container_number"].dropna().astype(str).unique().tolist())

    if hot_containers:
        return f"BL {bl} is HOT on container(s): {', '.join(hot_containers)}. All related containers: {', '.join(all_containers)}."
    else:
        return f"BL {bl} is not marked hot. Related containers: {', '.join(all_containers)}."


def _normalize_booking_token(s: str) -> str:
    """Normalize a booking token for comparison: uppercase alphanumerics only."""
    if s is None:
        return ""
    s = str(s).strip().upper()
    return re.sub(r"[^A-Z0-9]", "", s)

def _booking_in_cell(cell: str, booking_norm: str) -> bool:
    """Return True if normalized booking exists in a comma/sep-separated cell."""
    if pd.isna(cell) or not booking_norm:
        return False
    parts = re.split(r"[,;/\|\s]+", str(cell))
    for p in parts:
        if _normalize_booking_token(p) == booking_norm:
            return True
    return False

def _extract_booking_number(query: str) -> str | None:
    """
    Extract a booking number from free text.
    Examples: GT3000512, CN9140225, CN2229608
    """
    q = (query or "").strip().upper()
    if not q:
        return None

    # Prefer explicit "booking" phrasing
    m = re.search(r"\bBOOKING(?:\s+NUMBER)?\s*[:#-]?\s*([A-Z0-9\-\/]{6,20})\b", q, re.IGNORECASE)
    if m:
        cand = _normalize_booking_token(m.group(1))
        # Avoid capturing generic words like "NUMBER" or PO-like tokens
        if not re.search(r"\d", cand):
            return None
        if re.fullmatch(r"PO\d{5,12}", cand):
            return None
        if re.fullmatch(r"[A-Z]{4}\d{7}", cand):
            return None
        return cand

    # Fallback: common booking-like token patterns (2-4 letters + 5-10 digits)
    m2 = re.search(r"\b([A-Z]{2,4}\d{5,10})\b", q)
    if m2:
        cand = _normalize_booking_token(m2.group(1))
        # Exclude container IDs (AAAA#######) and PO-prefixed tokens
        if re.fullmatch(r"[A-Z]{4}\d{7}", cand):
            return None
        if re.fullmatch(r"PO\d{5,12}", cand):
            return None
        return cand

    # Last fallback: any single alnum token 6-20.
    # Guardrails: require at least one digit; exclude container pattern and common stopwords.
    tokens = re.findall(r"\b[A-Z0-9]{6,20}\b", q)
    for t in tokens:
        if not re.search(r"\d", t):
            continue
        if re.fullmatch(r"[A-Z]{4}\d{7}", t):  # container pattern
            continue
        if re.fullmatch(r"PO\d{5,12}", t):
            continue
        # Avoid returning generic tokens like BOOKING/NUMBER/CONTAINER in mixed text
        if t in {"BOOKING", "NUMBER", "CONTAINER", "PURCHASE", "ORDER", "SUPPLIER", "SHIPPER", "CONSIGNEE"}:
            continue
        return _normalize_booking_token(t)

    return None

def get_eta_for_booking(question: str = None, consignee_code: str = None, **kwargs):
    """
    Get ETA/arrival status for a Booking Number using booking_number_multiple (comma-separated).
    ETA preference: revised_eta > eta_dp.
    Arrival logic:
      - If ata_dp not null -> arrived (use ata_dp)
      - Else if derived_ata_dp not null:
          * if derived_ata_dp <= today -> treat as arrived on derived_ata_dp
          * else -> expected on derived_ata_dp
      - Else -> expected on preferred ETA (revised_eta > eta_dp) if available
    Returns list[dict] records (up to 50).
    """

    q = (question or kwargs.get("query") or kwargs.get("input") or "").strip()
    booking = _extract_booking_number(q)
    if not booking:
        return "Please specify a valid booking number (e.g., GT3000512)."

    df = _df().copy()
    if df is None or getattr(df, "empty", True):
        return "No shipment records available for your authorized consignees."

    # Optional explicit consignee filter (in addition to thread-local scoping)
    if consignee_code and "consignee_code_multiple" in df.columns:
        codes = [c.strip().upper() for c in str(consignee_code).split(",") if c.strip()]
        if codes:
            pat = r"|".join([re.escape(c) for c in codes])
            df = df[df["consignee_code_multiple"].astype(str).str.upper().str.contains(pat, na=False)].copy()
            if df.empty:
                return "No records found for provided consignee code(s)."

    if "booking_number_multiple" not in df.columns:
        return "Booking number column (booking_number_multiple) not found in the dataset."

    booking_norm = _normalize_booking_token(booking)

    mask = df["booking_number_multiple"].apply(lambda cell: _booking_in_cell(cell, booking_norm))
    matches = df[mask].copy()
    if matches.empty:
        return f"No data found for booking number {booking}."

    # Parse relevant dates
    date_cols = [c for c in ["revised_eta", "eta_dp", "ata_dp", "derived_ata_dp"] if c in matches.columns]
    if date_cols:
        matches = ensure_datetime(matches, date_cols)

    # Preferred ETA: revised_eta > eta_dp
    if "revised_eta" in matches.columns and "eta_dp" in matches.columns:
        matches["_eta_preferred"] = matches["revised_eta"].combine_first(matches["eta_dp"])
    elif "revised_eta" in matches.columns:
        matches["_eta_preferred"] = matches["revised_eta"]
    elif "eta_dp" in matches.columns:
        matches["_eta_preferred"] = matches["eta_dp"]
    else:
        matches["_eta_preferred"] = pd.NaT

    today = pd.Timestamp.today().normalize()

    def _status_row(r):
        ata = r.get("ata_dp", pd.NaT)
        d_ata = r.get("derived_ata_dp", pd.NaT)
        eta = r.get("_eta_preferred", pd.NaT)

        if pd.notna(ata):
            return "ARRIVED", ata
        if pd.notna(d_ata):
            if pd.Timestamp(d_ata).normalize() <= today:
                return "ARRIVED (DERIVED)", d_ata
            return "IN TRANSIT (EXPECTED DERIVED)", d_ata
        if pd.notna(eta):
            return "IN TRANSIT (EXPECTED)", eta
        return "UNKNOWN", pd.NaT

    tmp = matches.apply(lambda r: _status_row(r), axis=1, result_type="expand")
    tmp.columns = ["status", "_status_date"]
    matches["status"] = tmp["status"]
    matches["_status_date"] = tmp["_status_date"]

    # Sort: earliest relevant date first
    matches["_sort_date"] = matches["_status_date"].combine_first(matches["_eta_preferred"])
    matches = safe_sort_dataframe(matches, "_sort_date", ascending=True)

    out_cols = [
        "booking_number_multiple",
        "container_number",
        "po_number_multiple",
        "discharge_port",
        "revised_eta",
        "eta_dp",
        "derived_ata_dp",
        "ata_dp",
        "status",
        "_eta_preferred",
        "_status_date",
        "consignee_code_multiple",
    ]
    out_cols = [c for c in out_cols if c in matches.columns]
    out = matches[out_cols].head(50).copy()

    # Format date columns
    for dcol in ["revised_eta", "eta_dp", "derived_ata_dp", "ata_dp", "_eta_preferred", "_status_date"]:
        if dcol in out.columns and pd.api.types.is_datetime64_any_dtype(out[dcol]):
            out[dcol] = out[dcol].dt.strftime("%Y-%m-%d")

    out = out.rename(columns={"_eta_preferred": "eta", "_status_date": "status_date"})
    return out.where(pd.notnull(out), None).to_dict(orient="records")


def get_booking_details(question: str = None, consignee_code: str = None, **kwargs):
    """
    Booking-related lookup tool:
    - "booking number of PO <po>" / "booking number of <po>" -> booking_number_multiple
    - "booking number of container <container>" / "booking number of <container>" -> booking_number_multiple
    - "booking number of BL/OBL <bl>" / "booking number of <bl>" -> booking_number_multiple
    - "ETA of booking <booking>" -> delegates to get_eta_for_booking()
    - "PO of booking <booking>" / "container of booking <booking>" / "BL of booking <booking>" -> mapping outputs
    Notes:
    - booking_number_multiple is comma-separated.
    - Uses _df() (thread-local consignee scoping) + optional consignee_code param filtering.
    """

    q = (question or kwargs.get("query") or kwargs.get("input") or "").strip()
    if not q:
        return "Please provide a booking/PO/container/BL query."

    try:
        logger.info(f"[get_booking_details] Received query: {q!r}")
    except:
        pass

    q_lower = q.lower()
    q_upper = q.upper()

    # --- Remove injected/metadata-like prefixes so we don't mis-extract codes as POs ---
    q_extract = re.sub(
        r"\b(?:for\s+)?(?:consignee|user)(?:\s+code)?\s+\d{5,10}\b",
        "",
        q,
        flags=re.IGNORECASE,
    ).strip()

    # Extract identifiers
    container_no = extract_container_number(q_extract)
    if not container_no:
        m_cont = re.search(r"\b([A-Z]{4}\d{7})\b", q_upper)
        container_no = m_cont.group(1) if m_cont else None

    po_no = extract_po_number(q_extract)
    if not po_no:
        # Try explicit "PO" prefix
        m_po = re.search(r"\bPO\s*#?\s*(\d{5,})\b", q_upper)
        po_no = m_po.group(1) if m_po else None
    if not po_no:
        # **CRITICAL FIX**: If query has "booking number of <digits>", treat as PO
        if re.search(r"\bbooking\s+number\s+of\b", q_lower):
            m_numeric = re.search(r"\b(\d{7,})\b", q_upper)
            if m_numeric:
                po_no = m_numeric.group(1)

    try:
        obl_no = extract_ocean_bl_number(q_extract)
    except Exception:
        obl_no = None

    # **CRITICAL FIX**: Extract booking number more reliably
    booking_no = None

    # Pattern 1: Explicit "booking number X" or "booking X"
    m_booking = re.search(r'\b(?:booking\s+(?:number\s+)?|booking\s+#\s*)([A-Z0-9]{6,20})\b', q_upper)
    if m_booking:
        booking_no = m_booking.group(1)

    # Pattern 2: "of booking X" (for queries like "PO of booking VN2084805")
    if not booking_no:
        m_of_booking = re.search(r'\bof\s+booking\s+(?:number\s+)?([A-Z0-9]{6,20})\b', q_upper)
        if m_of_booking:
            booking_no = m_of_booking.group(1)

    # Pattern 3: Fallback - only extract if query contains "booking" keyword
    if not booking_no and re.search(r"\bbooking\b", q_lower):
        booking_no = _extract_booking_number(q_extract)

    # Agent/tool retries often pass a bare booking token (e.g., "VN2084805") without the word "booking".
    # If the input isn't clearly a PO/container/BL, treat booking-like tokens as booking numbers.
    if not booking_no and not (po_no or container_no or obl_no):
        booking_no = _extract_booking_number(q_extract)

    # Guard: if BL extractor grabbed a container-like token, prefer container semantics
    if obl_no and re.fullmatch(r"[A-Z]{4}\d{7}", str(obl_no).upper().strip()):
        obl_no = None

    try:
        logger.info(f"[get_booking_details] Extracted identifiers: container={container_no}, po={po_no}, obl={obl_no}, booking={booking_no}")
    except:
        pass

    # Intent flags
    wants_booking_number = bool(re.search(r"\bbooking\s+number\b", q_lower)) and not booking_no
    wants_eta = bool(re.search(r"\beta\b", q_lower)) and booking_no
    wants_po_of_booking = bool(re.search(r"\bpo\b", q_lower) and booking_no)
    wants_container_of_booking = bool(re.search(r"\bcontainer\b", q_lower) and booking_no)
    wants_bl_of_booking = bool(re.search(r"\b(?:bl|obl|bill\s+of\s+lading)\b", q_lower) and booking_no)

    # **CRITICAL FIX**: If user asks "X of booking Y", set the appropriate flag
    if booking_no and not (wants_eta or wants_po_of_booking or wants_container_of_booking or wants_bl_of_booking):
        if re.search(r'\bpo\s+of\s+booking\b', q_lower):
            wants_po_of_booking = True
        elif re.search(r'\bcontainer\s+of\s+booking\b', q_lower):
            wants_container_of_booking = True
        elif re.search(r'\b(?:bl|obl)\s+of\s+booking\b', q_lower):
            wants_bl_of_booking = True

    # If caller passes a bare identifier (PO/container/BL) without intent words,
    # interpret it as "booking number lookup" so agent/tool retries succeed.
    if (container_no or po_no or obl_no) and not (wants_eta or wants_po_of_booking or wants_container_of_booking or wants_bl_of_booking):
        wants_booking_number = True

    # 1) "ETA of booking <X>" -> delegate to existing ETA tool
    if booking_no and wants_eta:
        return get_eta_for_booking(question=q_extract, consignee_code=consignee_code)

    df = _df().copy()
    if df is None or getattr(df, "empty", True):
        return "No shipment records available for your authorized consignees."

    # Optional explicit consignee filter (in addition to thread-local scoping)
    if consignee_code and "consignee_code_multiple" in df.columns:
        codes = [c.strip().upper() for c in str(consignee_code).split(",") if c.strip()]
        if codes:
            pat = r"|".join([re.escape(c) for c in codes])
            df = df[df["consignee_code_multiple"].astype(str).str.upper().str.contains(pat, na=False)].copy()
            if df.empty:
                return "No records found for provided consignee code(s)."

    # Agent retries sometimes pass only the raw identifier (e.g., "VN2084805")
    # without the word "booking". If it exists in booking_number_multiple, treat it
    # as a booking number so we can return PO/container/BL mapping in one call.
    if not booking_no and "booking_number_multiple" in df.columns:
        m_code = re.search(r"\b[A-Z0-9]{6,20}\b", q_upper)
        if m_code:
            cand = m_code.group(0)
            if not re.fullmatch(r"\d{5,}", cand) and not re.fullmatch(r"[A-Z]{4}\d{7}", cand):
                cand_norm = _normalize_booking_token(cand)
                try:
                    mask_cand = df["booking_number_multiple"].apply(lambda cell: _booking_in_cell(cell, cand_norm))
                    if mask_cand.any():
                        booking_no = cand
                except Exception:
                    pass

    if "booking_number_multiple" not in df.columns:
        return "Booking number column (booking_number_multiple) not found in the dataset."

    # If a booking number was provided (or inferred) but the user/agent didn't specify
    # PO/container/BL explicitly, default to returning the mapping for that booking.
    # This avoids multi-turn retries like: tool_input="VN2084805" -> "booking number VN2084805" -> ...
    if booking_no and not (wants_eta or wants_po_of_booking or wants_container_of_booking or wants_bl_of_booking):
        wants_po_of_booking = True

    # A) "PO/container/BL of booking <booking>" - **ENHANCED**
    if booking_no and (wants_po_of_booking or wants_container_of_booking or wants_bl_of_booking):
        booking_norm = _normalize_booking_token(booking_no)
        mask = df["booking_number_multiple"].apply(lambda cell: _booking_in_cell(cell, booking_norm))
        matches = df[mask].copy()
        if matches.empty:
            return f"No data found for booking {booking_no}."

        out_cols = [
            "booking_number_multiple",
            "container_number",
            "po_number_multiple",
            "po_number",
            "ocean_bl_no_multiple",
            "discharge_port",
            "consignee_code_multiple",
        ]
        # include ETA/arrival context if present
        for c in ["revised_eta", "eta_dp", "ata_dp", "derived_ata_dp"]:
            if c in matches.columns:
                out_cols.append(c)

        out_cols = [c for c in out_cols if c in matches.columns]
        out = matches[out_cols].drop_duplicates().head(100).copy()

        # Format dates if present
        date_cols = [c for c in ["revised_eta", "eta_dp", "ata_dp", "derived_ata_dp"] if c in out.columns]
        if date_cols:
            out = ensure_datetime(out, date_cols)
            for c in date_cols:
                if pd.api.types.is_datetime64_any_dtype(out[c]):
                    out[c] = out[c].dt.strftime("%Y-%m-%d")

        return out.where(pd.notnull(out), None).to_dict(orient="records")

    # B) "booking number of PO/container/BL"
    if wants_booking_number:
        matches = pd.DataFrame()

        # Priority: container > PO > BL
        if container_no:
            if "container_number" not in df.columns:
                return "Container number column not found in the dataset."
            clean = clean_container_number(container_no)
            cont_norm = df["container_number"].astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
            matches = df[cont_norm == clean].copy()
            if matches.empty:
                matches = df[df["container_number"].astype(str).str.contains(container_no, case=False, na=False)].copy()
            if matches.empty:
                # **FALLBACK**: Maybe it's actually a PO misidentified as container
                if re.match(r"^\d{7,}$", container_no):
                    po_col = "po_number_multiple" if "po_number_multiple" in df.columns else "po_number"
                    if po_col in df.columns:
                        po_norm = _normalize_po_token(container_no)
                        mask = df[po_col].apply(lambda cell: _po_in_cell(cell, po_norm))
                        matches = df[mask].copy()
                        if not matches.empty:
                            pass  # Found as PO, continue
                        else:
                            return f"No booking found for container or PO {container_no}."
                    else:
                        return f"No data found for container {container_no}."
                else:
                    return f"No data found for container {container_no}."

        elif po_no:
            po_col = "po_number_multiple" if "po_number_multiple" in df.columns else ("po_number" if "po_number" in df.columns else None)
            if not po_col:
                return "PO column not found in the dataset."
            po_norm = _normalize_po_token(po_no)
            mask = df[po_col].apply(lambda cell: _po_in_cell(cell, po_norm))
            matches = df[mask].copy()
            if matches.empty:
                return f"No data found for PO {po_no}."

        elif obl_no:
            bl_col = _find_ocean_bl_col(df) or ("ocean_bl_no_multiple" if "ocean_bl_no_multiple" in df.columns else None)
            if not bl_col:
                return "Ocean BL column not found in the dataset."
            bl_norm = _normalize_bl_token(obl_no)
            mask = df[bl_col].apply(lambda cell: _bl_in_cell(cell, bl_norm))
            matches = df[mask].copy()
            if matches.empty:
                return f"No data found for OBL/BL {obl_no}."

        else:
            # **ENHANCED FALLBACK**: if user wrote "booking number of 5300008696" (no PO prefix)
            # treat as PO-like numeric token if present and not a container/booking token
            m_num = re.search(r"\b(\d{7,})\b", q_upper)  # POs are typically 7+ digits
            if m_num:
                po_guess = m_num.group(1)
                po_col = "po_number_multiple" if "po_number_multiple" in df.columns else ("po_number" if "po_number" in df.columns else None)
                if po_col:
                    po_norm = _normalize_po_token(po_guess)
                    mask = df[po_col].apply(lambda cell: _po_in_cell(cell, po_norm))
                    matches = df[mask].copy()
                    if matches.empty:
                        # **SECONDARY FALLBACK**: Try simple contains match
                        matches = df[df[po_col].astype(str).str.upper().str.contains(po_guess, na=False)].copy()
                    if matches.empty:
                        return f"No booking found for PO {po_guess}. Please verify the PO number."
                else:
                    return "PO column not found in the dataset."
            else:
                return "Please specify a PO number, container number, or ocean BL/OBL to look up the booking number."

        out_cols = [
            "booking_number_multiple",
            "container_number",
            "po_number_multiple",
            "po_number",
            "ocean_bl_no_multiple",
            "discharge_port",
            "consignee_code_multiple",
        ]
        out_cols = [c for c in out_cols if c in matches.columns]
        out = matches[out_cols].drop_duplicates().head(100).copy()
        return out.where(pd.notnull(out), None).to_dict(orient="records")

    return "Please ask a booking-related question (booking number of PO/container/BL, ETA of booking, or PO/container/BL of booking)."


################### fd related query strated from here #########################
def get_containers_by_final_destination(query: str) -> str:
    """
    Find containers arriving at a specific final destination/distribution center within a specified time period.
    - Handles queries like "list containers arriving at Nashville in next 3 days" or "containers to DC Phoenix next week"
    - Phrase synonyms: fd, in-dc, in dc, final destination, distribution center, warehouse, terminal
    - Use final_destination column for filtering
    - Reached/Arrived: Include rows where delivery_date_to_consignee is NOT null
    - Delay calculation: delay => delivery_date_to_consignee - eta_fd
    - Early calculation: (- delay) i.e. delay < 0
    - Delay period: take from user query if mentioned else default = 7 days
    - Delay/Early filter: where delivery_date_to_consignee != null AND delay
        - by or less that filter: delay < delay_period
        - more than filter: delay > delay_period
    - `del_date` date priority: predictive_eta_fd -> revised_eta_fd -> eta_fd
    - Arriving/Coming: delivery_date_to_consignee == null AND del_date between current date and current_date + time_delta
    - Per-row ETA selection: use revised_eta if present, otherwise eta_fd
    - Hot containers filter: Use hot_container_flag column
    - Transport mode filter: Use transport_mode column
    - consignee Filterization: filter only based on consignee_name, which will get from user query

    Returns list[dict] with container, po, obl columns  AND Columns names related to Operation performed on
    """
    # Log the incoming query for debugging
    try:
        logger.info(f"[get_containers_by_final_destination] Received query: {query!r}")
    except:
        print(f"[get_containers_by_final_destination] Received query: {query!r}")

    # Parse days
    default_days = 7
    days = None
    for pat in [
        r"(?:next|upcoming|within|in)\s+(\d{1,3})\s+days?",
        r"arriving.*?(\d{1,3})\s+days?",
        r"(\d{1,3})\s+days?"
    ]:
        m = re.search(pat, query, re.IGNORECASE)
        if m:
            days = int(m.group(1))
            break
    n_days = days if days is not None else default_days
    logger.info(f"\n days: {n_days}")

    today = pd.Timestamp.today().normalize()
    end_date = today + pd.Timedelta(days=n_days)

    # Log parsed timeframe for debugging
    try:
        logger.info(f"[get_containers_by_final_destination] parsed_n_days={n_days} today={today.strftime('%Y-%m-%d')} end_date={end_date.strftime('%Y-%m-%d')}")
    except Exception:
        print(f"[get_containers_by_final_destination] parsed_n_days={n_days} today={today} end_date={end_date}")

    df = _df()  # respects consignee filtering

    # # Check if final_destination column exists
    # if 'final_destination' not in df.columns:
    #     return "No final_destination column found in the dataset."

    # Extract destination from query using various patterns
    location_patterns = [
        r'(?:at|to|in)\s+(?:fd|dc|final\s+destination|distribution\s+center)\s+([A-Za-z\s\.]{3,}?)(?:[,\s]|$)',  # "fd Nashville", "dc Phoenix"
        r'(?:fd|dc|final\s+destination|distribution\s+center)\s+(?:at|to|in)\s+([A-Za-z\s\.]{3,}?)(?:[,\s]|$)',  # "fd at Nashville"
        r'(?:at|to|in)\s+([A-Za-z\s\.]{3,}?)(?:\s+fd|\s+dc|\s+final\s+destination|\s+distribution\s+center)(?:[,\s]|$)',  # "at Nashville fd"
        # r'(?:at|to|in)\s+([A-Za-z\s\.]{3,}?)(?:[,\s]|$)'  # fallback: "at Nashville"
    ]

    destination_name = None
    for pattern in location_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            destination_name = match.group(1).strip()
            break

    logger.info(f"Location = {destination_name}")
    filtered_df = df.copy()
    if destination_name:
        # return "Please specify a final destination or distribution center."
        # Filter by destination
        destination_mask = df['final_destination'].astype(str).str.upper().str.contains(destination_name.upper(), na=False)
        filtered_df = df[destination_mask].copy()

    if destination_name and filtered_df.empty:
        return f"No containers found with final destination containing '{destination_name}' for your authorized consignees."

    # determine per-row ETA using revised_eta then eta_dp
    date_priority = [c for c in ['predictive_eta_fd', 'revised_eta_fd', 'eta_fd'] if c in filtered_df.columns]
    if not date_priority:
        return "No ETA columns (predictive_eta_fd-> revised_eta_fd-> eta_fd) found in the data to compute upcoming arrivals."

    parse_cols = date_priority.copy()
    if 'delivery_date_to_consignee' in filtered_df.columns:
        parse_cols.append('delivery_date_to_consignee')

    filtered_df = ensure_datetime(filtered_df, parse_cols)

    filtered_df['eta_for_filter'] = filtered_df['predictive_eta_fd'].combine_first(filtered_df['revised_eta_fd']).combine_first(filtered_df['eta_fd'])

    # filter: eta_for_filter between today..end_date and ata_dp is null (not arrived)
    date_mask = (filtered_df['eta_for_filter'] >= today) & (filtered_df['eta_for_filter'] <= end_date)
    if 'delivery_date_to_consignee' in filtered_df.columns:
        date_mask &= filtered_df['delivery_date_to_consignee'].isna()

    result = filtered_df[date_mask].copy()
    if result.empty:
        return f"No containers arriving at final destination '{destination_name}' between {today.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')} for your authorized consignees."

    # prepare output columns and format dates
    out_cols = ['container_number', 'po_number_multiple', 'final_destination', 'revised_eta_fd', 'eta_fd', 'eta_for_filter']
    out_cols = [c for c in out_cols if c in result.columns]

    out_df = result[out_cols].sort_values('eta_for_filter').head(50).copy()

    for d in ['revised_eta', 'eta_dp', 'eta_for_filter']:
        if d in out_df.columns and pd.api.types.is_datetime64_any_dtype(out_df[d]):
            out_df[d] = out_df[d].dt.strftime('%Y-%m-%d')

    if 'eta_for_filter' in out_df.columns:
        out_df = out_df.drop(columns=['eta_for_filter'])

    return out_df.where(pd.notnull(out_df), None).to_dict(orient='records')

######################### fd related query ended here ##########################


def get_eta_for_po(question: str = None, consignee_code: str = None, **kwargs) -> str:
    """
    Get ETA for a PO (handles 'PO6300134648', 'po 6300134648', or '6300134648').
    - Matches against po_number_multiple (comma-separated) or po_number.
    - Prefers revised_eta over eta_dp.
    - Respects thread-local consignee filtering via _df(); optional explicit consignee_code filter supported.
    Returns list[dict]: [po_number_multiple/po_number, container_number, discharge_port, revised_eta, eta_dp].
    """


    q = (question or "").strip()
    if not q:
        return "Please specify a PO number."

    # Extract PO tokens (helper + robust fallback)
    try:
        base_po = extract_po_number(q)
    except Exception:
        base_po = None

    tokens = set()
    if base_po:
        tokens.add(base_po)
    for t in re.findall(r'\b[A-Z]*[-#]?\d{6,}[A-Z]*\b', q.upper()):
        tokens.add(re.sub(r'[^A-Z0-9]', '', t))
    if not tokens:
        return "Please specify a valid PO number."

    po_norms = {_normalize_po_token(t) for t in tokens if t}
    if not po_norms:
        return "Please specify a valid PO number."

    df = _df().copy()
    if df.empty:
        return "No PO records found for your authorized consignees."

    # Optional explicit consignee filter
    if consignee_code and "consignee_code_multiple" in df.columns:
        codes = [c.strip().upper() for c in str(consignee_code).split(",") if c.strip()]
        if codes:
            pat = r"|".join([re.escape(c) for c in codes])
            df = df[df["consignee_code_multiple"].astype(str).str.upper().str.contains(pat, na=False)].copy()
            if df.empty:
                return "No PO records found for provided consignee codes."

    # Choose PO column
    po_col = "po_number_multiple" if "po_number_multiple" in df.columns else ("po_number" if "po_number" in df.columns else None)
    if not po_col:
        return "PO column not found in the dataset."

    # Match rows where any normalized token matches (handles comma-separated)
    def row_has_any_po(cell: str) -> bool:
        if pd.isna(cell):
            return False
        return any(_po_in_cell(cell, tok) for tok in po_norms)

    mask = df[po_col].apply(row_has_any_po)
    matches = df[mask].copy()
    if matches.empty:
        rep = next(iter(po_norms))
        return f"No data found for PO {rep}."

    # Parse dates, prefer revised_eta over eta_dp
    date_cols = [c for c in ["revised_eta", "eta_dp", "ata_dp"] if c in matches.columns]
    if date_cols:
        matches = ensure_datetime(matches, date_cols)

    # Sort by preferred ETA (revised_eta first, else eta_dp), nulls last
    if "revised_eta" in matches.columns and "eta_dp" in matches.columns:
        matches["_eta_for_sort"] = matches["revised_eta"].where(matches["revised_eta"].notna(), matches["eta_dp"])
    elif "revised_eta" in matches.columns:
        matches["_eta_for_sort"] = matches["revised_eta"]
    elif "eta_dp" in matches.columns:
        matches["_eta_for_sort"] = matches["eta_dp"]

    cols = [po_col, "container_number", "discharge_port", "revised_eta", "eta_dp"]
    cols = [c for c in cols if c in matches.columns]
    out = matches[cols + (["_eta_for_sort"] if "_eta_for_sort" in matches.columns else [])].copy()
    if "_eta_for_sort" in out.columns:
        out = out.sort_values("_eta_for_sort", na_position="last").drop(columns=["_eta_for_sort"])

    # Format dates
    for d in ["revised_eta", "eta_dp"]:
        if d in out.columns and pd.api.types.is_datetime64_any_dtype(out[d]):
            out[d] = out[d].dt.strftime("%Y-%m-%d")

    return out.head(100).where(pd.notnull(out), None).to_dict(orient="records")


def get_container_transit_analysis(query: str) -> str:
    """
    Analyzes transit times, delays, and journey metrics for specific containers.

    Supports queries like:
    - "What is estimated transit time for container ABCD1234567?"
    - "Show actual transit time for WXYZ9876543"
    - "What is the delay for container TCLU4521258?"
    - "Transit analysis for container MRSU3925910"
    - "How long did container HASU5127798 take to arrive?"
    - "Journey time for container UACU8494735"
    - "Transit time for ABCD1234567 from Shanghai to Rotterdam"
    - "Show delay for WXYZ9876543 by Maersk"
    - "Transit analysis for TCLU4521258 in November"

    Returns detailed transit metrics including:
    - Estimated transit days (ETD_LP to ETA_DP)
    - Actual transit days (ATD_LP to ATA_DP)
    - Delay days (ETA_DP vs ATA_DP)
    - Route information (Load Port to Discharge Port)
    - Key dates (ETD, ATD, ETA, ATA, Revised ETA)

    Supports additional filters:
    - Load port filtering: "from Shanghai", "from CNSHA"
    - Discharge port filtering: "to Rotterdam", "to NLRTM"
    - Carrier filtering: "by Maersk", "carrier MSC"
    - Transport mode filtering: "by sea", "by air"
    - Date range filtering: "in November", "last month", "between dates"
    """

    query = (query or "").strip()
    query_upper = query.upper()
    query_lower = query.lower()

    # ========== 1) EXTRACT CONTAINER NUMBER ==========
    container_no = extract_container_number(query)

    if not container_no:
        return "Please provide a valid container number to analyze transit time."

    try:
        logger.info(f"[get_container_transit_analysis] Analyzing container: {container_no}")
    except:
        pass

    # ========== 2) EXTRACT ADDITIONAL FILTERS ==========
    # Extract load port
    load_port = None
    for pattern in [
        r'from\s+(?:load\s+port\s+)?([A-Z][A-Z\s,\-\(\)]+?)(?=\s+to\s+|\s+by\s+|\s+in\s+|$)',
        r'load\s+port\s+([A-Z][A-Z\s,\-\(\)]+?)(?=\s+to\s+|\s+by\s+|$)',
        r'origin\s+([A-Z][A-Z\s,\-\(\)]+?)(?=\s+to\s+|\s+by\s+|$)'
    ]:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            load_port = match.group(1).strip().upper()
            break

    # Extract discharge port
    discharge_port = None
    for pattern in [
        r'to\s+(?:discharge\s+port\s+)?([A-Z][A-Z\s,\-\(\)]+?)(?=\s+by\s+|\s+in\s+|$)',
        r'discharge\s+port\s+([A-Z][A-Z\s,\-\(\)]+?)(?=\s+by\s+|\s+in\s+|$)',
        r'destination\s+([A-Z][A-Z\s,\-\(\)]+?)(?=\s+by\s+|\s+in\s+|$)'
    ]:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            discharge_port = match.group(1).strip().upper()
            break

    # Extract carrier
    carrier = None
    match = re.search(r'(?:carrier|by)\s+([A-Z][A-Z\s]+?)(?=\s+in\s+|\s+from\s+|\s+to\s+|$)', query, re.IGNORECASE)
    if match:
        carrier = match.group(1).strip().upper()

    # Extract transport mode
    modes = extract_transport_modes(query)

    # Extract date range (if mentioned)
    date_filter_start = None
    date_filter_end = None
    date_desc = None
    if re.search(r'\b(in|during|for|last|this|next)\s+(january|february|march|april|may|june|july|august|september|october|november|december|month|week|year)', query, re.IGNORECASE):
        try:
            from agents.prompts import parse_time_period
            date_filter_start, date_filter_end, date_desc = parse_time_period(query)
        except Exception:
            pass

    try:
        logger.info(f"[get_container_transit_analysis] Filters: load_port={load_port}, discharge_port={discharge_port}, carrier={carrier}, modes={modes}, dates={date_desc}")
    except:
        pass

    # ========== 3) LOAD DATA ==========
    df = _df()  # Respects consignee filtering

    if df.empty:
        return "No data available for your authorized consignees."

    # ========== 4) FILTER BY CONTAINER NUMBER ==========
    if 'container_number' not in df.columns:
        return "Container number column not found in the dataset."

    clean_cont = clean_container_number(container_no)
    cont_col_norm = df["container_number"].astype(str).str.replace(r'[^A-Z0-9]', '', regex=True)
    container_mask = cont_col_norm == clean_cont

    if not container_mask.any():
        container_mask = df["container_number"].astype(str).str.contains(container_no, case=False, na=False)

    results = df[container_mask].copy()

    if results.empty:
        return f"Container {container_no} not found in the dataset."

    try:
        logger.info(f"[get_container_transit_analysis] Found {len(results)} record(s) for container {container_no}")
    except:
        pass

    # ========== 5) APPLY ADDITIONAL FILTERS ==========
    filters_applied = []

    # Load port filter
    if load_port and 'load_port' in results.columns:
        results = results[results['load_port'].astype(str).str.contains(load_port, case=False, na=False)]
        filters_applied.append(f"load port: {load_port}")
        if results.empty:
            return f"Container {container_no} not found with load port {load_port}."

    # Discharge port filter
    if discharge_port and 'discharge_port' in results.columns:
        results = results[results['discharge_port'].astype(str).str.contains(discharge_port, case=False, na=False)]
        filters_applied.append(f"discharge port: {discharge_port}")
        if results.empty:
            return f"Container {container_no} not found with discharge port {discharge_port}."

    # Carrier filter
    if carrier and 'final_carrier_name' in results.columns:
        results = results[results['final_carrier_name'].astype(str).str.contains(carrier, case=False, na=False)]
        filters_applied.append(f"carrier: {carrier}")
        if results.empty:
            return f"Container {container_no} not found with carrier {carrier}."

    # Transport mode filter
    if modes and 'transport_mode' in results.columns:
        results = results[results['transport_mode'].astype(str).str.lower().apply(lambda s: any(m in s for m in modes))]
        filters_applied.append(f"transport mode: {', '.join(modes)}")
        if results.empty:
            return f"Container {container_no} not found with transport mode {', '.join(modes)}."

    try:
        if filters_applied:
            logger.info(f"[get_container_transit_analysis] Applied filters: {', '.join(filters_applied)}")
    except:
        pass

    # ========== 6) PARSE DATE COLUMNS ==========
    date_cols = ['etd_lp', 'atd_lp', 'eta_dp', 'ata_dp', 'revised_eta', 'predictive_eta']
    existing_date_cols = [c for c in date_cols if c in results.columns]

    if existing_date_cols:
        results = ensure_datetime(results, existing_date_cols)

    # ========== 7) APPLY DATE RANGE FILTER (if specified) ==========
    if date_filter_start and date_filter_end and 'ata_dp' in results.columns:
        date_mask = (results['ata_dp'].notna()) & \
                   (results['ata_dp'].dt.normalize() >= date_filter_start) & \
                   (results['ata_dp'].dt.normalize() <= date_filter_end)
        results = results[date_mask]
        filters_applied.append(f"date range: {date_desc}")
        if results.empty:
            return f"Container {container_no} has no arrival records in {date_desc}."

    # ========== 8) CALCULATE TRANSIT METRICS ==========
    transit_metrics = []

    for idx, row in results.iterrows():
        metrics = {
            'container_number': row.get('container_number', 'N/A'),
            'load_port': row.get('load_port', 'N/A'),
            'discharge_port': row.get('discharge_port', 'N/A'),
        }

        # Extract dates
        etd_lp = row.get('etd_lp')
        atd_lp = row.get('atd_lp')
        eta_dp = row.get('eta_dp')
        ata_dp = row.get('ata_dp')
        revised_eta = row.get('revised_eta')

        # Add date values
        metrics['etd_lp'] = etd_lp.strftime('%Y-%m-%d') if pd.notna(etd_lp) else None
        metrics['atd_lp'] = atd_lp.strftime('%Y-%m-%d') if pd.notna(atd_lp) else None
        metrics['eta_dp'] = eta_dp.strftime('%Y-%m-%d') if pd.notna(eta_dp) else None
        metrics['ata_dp'] = ata_dp.strftime('%Y-%m-%d') if pd.notna(ata_dp) else None
        metrics['revised_eta'] = revised_eta.strftime('%Y-%m-%d') if pd.notna(revised_eta) else None

        # ── ESTIMATED TRANSIT TIME ──
        # Use pre-calculated column if available, otherwise calculate
        if 'estimated_transit_days' in row.index and pd.notna(row['estimated_transit_days']):
            metrics['estimated_transit_days'] = int(row['estimated_transit_days'])
        elif pd.notna(etd_lp) and pd.notna(eta_dp):
            estimated_transit = (eta_dp - etd_lp).days
            metrics['estimated_transit_days'] = estimated_transit
        else:
            metrics['estimated_transit_days'] = None

        # ── ACTUAL TRANSIT TIME ──
        # Use pre-calculated column if available, otherwise calculate
        if 'actual_transit_days' in row.index and pd.notna(row['actual_transit_days']):
            metrics['actual_transit_days'] = int(row['actual_transit_days'])
        elif pd.notna(atd_lp) and pd.notna(ata_dp):
            actual_transit = (ata_dp - atd_lp).days
            metrics['actual_transit_days'] = actual_transit
        elif pd.notna(etd_lp) and pd.notna(ata_dp):
            # Fallback: use ETD if ATD not available
            actual_transit = (ata_dp - etd_lp).days
            metrics['actual_transit_days'] = actual_transit
            metrics['note'] = "Actual transit calculated using ETD (ATD not available)"
        else:
            metrics['actual_transit_days'] = None

        # ── DELAY CALCULATION (ETA_DP vs ATA_DP) ──
        if pd.notna(eta_dp) and pd.notna(ata_dp):
            delay = (ata_dp - eta_dp).days
            metrics['delay_days'] = delay

            # Status based on delay
            if delay > 0:
                metrics['delivery_status'] = f"Delayed by {delay} days"
            elif delay < 0:
                metrics['delivery_status'] = f"Early by {abs(delay)} days"
            else:
                metrics['delivery_status'] = "On time"
        elif pd.notna(eta_dp) and pd.isna(ata_dp):
            # Container not yet arrived
            today = pd.Timestamp.today().normalize()
            if eta_dp < today:
                days_overdue = (today - eta_dp).days
                metrics['delay_days'] = days_overdue
                metrics['delivery_status'] = f"Overdue by {days_overdue} days (not yet arrived)"
            else:
                metrics['delay_days'] = None
                metrics['delivery_status'] = "In transit (not yet arrived)"
        else:
            metrics['delay_days'] = None
            metrics['delivery_status'] = "Unknown"

        # ── TRANSIT EFFICIENCY ──
        if metrics['estimated_transit_days'] and metrics['actual_transit_days']:
            efficiency = (metrics['estimated_transit_days'] / metrics['actual_transit_days']) * 100
            metrics['transit_efficiency_pct'] = round(efficiency, 2)
        else:
            metrics['transit_efficiency_pct'] = None

        # Add additional context
        metrics['final_carrier_name'] = row.get('final_carrier_name', 'N/A')
        metrics['transport_mode'] = row.get('transport_mode', 'N/A')

        # Hot container flag
        hot_flag_cols = [c for c in results.columns if 'hot_container_flag' in c.lower()]
        if hot_flag_cols:
            hot_val = row.get(hot_flag_cols[0])
            metrics['hot_container'] = str(hot_val).strip().upper() in {"Y", "YES", "TRUE", "1", "HOT"}
        else:
            metrics['hot_container'] = False

        # PO and BL information
        metrics['po_numbers'] = row.get('po_number_multiple', 'N/A')

        bl_col = _find_ocean_bl_col(results)
        if bl_col:
            metrics['ocean_bl_numbers'] = row.get(bl_col, 'N/A')

        transit_metrics.append(metrics)

    # ========== 9) FORMAT RESPONSE ==========
    if not transit_metrics:
        return f"No transit data available for container {container_no}."

    # If multiple records, take the most recent
    if len(transit_metrics) > 1:
        try:
            logger.info(f"[get_container_transit_analysis] Found {len(transit_metrics)} records, returning all")
        except:
            pass

    # ========== 10) PREPARE OUTPUT ==========
    output_cols = [
        'container_number',
        'load_port',
        'discharge_port',
        'etd_lp',
        'atd_lp',
        'eta_dp',
        'ata_dp',
        'revised_eta',
        'estimated_transit_days',
        'actual_transit_days',
        'delay_days',
        'delivery_status',
        'transit_efficiency_pct',
        'final_carrier_name',
        'transport_mode',
        'hot_container',
        'po_numbers',
        'ocean_bl_numbers'
    ]

    # Filter to available columns
    output_data = []
    for metric in transit_metrics:
        output_record = {k: metric.get(k) for k in output_cols if k in metric}
        output_data.append(output_record)

    try:
        logger.info(f"[get_container_transit_analysis] Returning {len(output_data)} transit analysis record(s)")
    except:
        pass

    return output_data


def get_po_transit_analysis(query: str) -> str:
    """
    Analyzes transit times for all containers associated with a specific PO.

    Supports queries like:
    - "Transit analysis for PO 6300134648"
    - "Show transit times for purchase order 5302816722"
    - "What is the transit performance for PO 6300134648?"
    - "Delay analysis for PO 5302816722"
    - "How are containers for PO 6300134648 performing?"

    Returns:
    - Individual transit metrics for each container in the PO
    - Summary statistics (average transit time, average delay, etc.)
    - Container count and completion status
    """

    query = (query or "").strip()

    # Extract PO number
    po_no = extract_po_number(query)
    if not po_no:
        return "Please provide a valid PO number to analyze transit time."

    try:
        logger.info(f"[get_po_transit_analysis] Analyzing PO: {po_no}")
    except:
        pass

    # Load data
    df = _df()
    if df.empty:
        return "No data available for your authorized consignees."

    # Find PO column
    po_col = "po_number_multiple" if "po_number_multiple" in df.columns else ("po_number" if "po_number" in df.columns else None)
    if not po_col:
        return "PO number column not found in the dataset."

    # Match PO
    po_norm = _normalize_po_token(po_no)
    mask = df[po_col].apply(lambda cell: _po_in_cell(cell, po_norm))
    results = df[mask].copy()

    if results.empty:
        return f"No containers found for PO {po_no}."

    try:
        logger.info(f"[get_po_transit_analysis] Found {len(results)} container(s) for PO {po_no}")
    except:
        pass

    # Parse dates
    date_cols = ['etd_lp', 'atd_lp', 'eta_dp', 'ata_dp', 'revised_eta']
    existing_date_cols = [c for c in date_cols if c in results.columns]
    if existing_date_cols:
        results = ensure_datetime(results, existing_date_cols)

    # Calculate transit metrics for each container
    container_metrics = []

    for idx, row in results.iterrows():
        metrics = {
            'container_number': row.get('container_number', 'N/A'),
            'po_number': po_no,
            'load_port': row.get('load_port', 'N/A'),
            'discharge_port': row.get('discharge_port', 'N/A'),
        }

        etd_lp = row.get('etd_lp')
        atd_lp = row.get('atd_lp')
        eta_dp = row.get('eta_dp')
        ata_dp = row.get('ata_dp')

        # Dates
        metrics['etd_lp'] = etd_lp.strftime('%Y-%m-%d') if pd.notna(etd_lp) else None
        metrics['atd_lp'] = atd_lp.strftime('%Y-%m-%d') if pd.notna(atd_lp) else None
        metrics['eta_dp'] = eta_dp.strftime('%Y-%m-%d') if pd.notna(eta_dp) else None
        metrics['ata_dp'] = ata_dp.strftime('%Y-%m-%d') if pd.notna(ata_dp) else None

        # Estimated transit - use pre-calculated if available
        if 'estimated_transit_days' in row.index and pd.notna(row['estimated_transit_days']):
            metrics['estimated_transit_days'] = int(row['estimated_transit_days'])
        elif pd.notna(etd_lp) and pd.notna(eta_dp):
            metrics['estimated_transit_days'] = (eta_dp - etd_lp).days
        else:
            metrics['estimated_transit_days'] = None

        # Actual transit - use pre-calculated if available
        if 'actual_transit_days' in row.index and pd.notna(row['actual_transit_days']):
            metrics['actual_transit_days'] = int(row['actual_transit_days'])
        elif pd.notna(atd_lp) and pd.notna(ata_dp):
            metrics['actual_transit_days'] = (ata_dp - atd_lp).days
        elif pd.notna(etd_lp) and pd.notna(ata_dp):
            metrics['actual_transit_days'] = (ata_dp - etd_lp).days
        else:
            metrics['actual_transit_days'] = None

        # Delay - use pre-calculated if available
        if 'delay_days' in row.index and pd.notna(row['delay_days']):
            delay = int(row['delay_days'])
            metrics['delay_days'] = delay
        elif pd.notna(eta_dp) and pd.notna(ata_dp):
            delay = (ata_dp - eta_dp).days
            metrics['delay_days'] = delay
            if delay > 0:
                metrics['status'] = f"Delayed by {delay} days"
            elif delay < 0:
                metrics['status'] = f"Early by {abs(delay)} days"
            else:
                metrics['status'] = "On time"
        elif pd.notna(eta_dp) and pd.isna(ata_dp):
            metrics['delay_days'] = None
            metrics['status'] = "In transit"
        else:
            metrics['delay_days'] = None
            metrics['status'] = "Unknown"

        metrics['carrier'] = row.get('final_carrier_name', 'N/A')
        metrics['transport_mode'] = row.get('transport_mode', 'N/A')

        container_metrics.append(metrics)

    # Calculate summary statistics
    transit_days = [m['actual_transit_days'] for m in container_metrics if m['actual_transit_days'] is not None]
    delay_days = [m['delay_days'] for m in container_metrics if m['delay_days'] is not None]

    summary = {
        'po_number': po_no,
        'total_containers': len(container_metrics),
        'arrived_containers': sum(1 for m in container_metrics if m.get('ata_dp') is not None),
        'in_transit_containers': sum(1 for m in container_metrics if m.get('ata_dp') is None),
        'avg_transit_days': round(sum(transit_days) / len(transit_days), 1) if transit_days else None,
        'avg_delay_days': round(sum(delay_days) / len(delay_days), 1) if delay_days else None,
        'delayed_containers': sum(1 for d in delay_days if d > 0),
        'on_time_or_early_containers': sum(1 for d in delay_days if d <= 0)
    }

    response = {
        'summary': summary,
        'container_details': container_metrics
    }

    try:
        logger.info(f"[get_po_transit_analysis] Returning analysis for {len(container_metrics)} containers")
    except:
        pass

    return [response]


def get_bl_transit_analysis(query: str) -> str:
    """
    Analyzes transit times for all containers associated with a specific Ocean BL.

    Supports queries like:
    - "Transit analysis for BL MOLWMNL2400017"
    - "Show transit times for ocean BL MOLWMNL2400017"
    - "What is the transit performance for bill of lading MOLWMNL2400017?"
    - "Delay analysis for BL MOLWMNL2400017"

    Returns:
    - Individual transit metrics for each container in the BL
    - Summary statistics
    - Container count and completion status
    """

    query = (query or "").strip()

    # Extract BL number
    try:
        bl_no = extract_ocean_bl_number(query)
    except:
        bl_no = None

    if not bl_no:
        # Fallback regex
        m = re.search(r'\b([A-Z0-9]{6,24})\b', query.upper())
        bl_no = m.group(1) if m else None

    if not bl_no:
        return "Please provide a valid ocean BL number to analyze transit time."

    try:
        logger.info(f"[get_bl_transit_analysis] Analyzing BL: {bl_no}")
    except:
        pass

    # Load data
    df = _df()
    if df.empty:
        return "No data available for your authorized consignees."

    # Find BL column
    bl_col = _find_ocean_bl_col(df)
    if not bl_col:
        return "Ocean BL column not found in the dataset."

    # Match BL
    bl_norm = _normalize_bl_token(bl_no)
    mask = df[bl_col].apply(lambda cell: _bl_in_cell(cell, bl_norm))
    results = df[mask].copy()

    if results.empty:
        return f"No containers found for ocean BL {bl_no}."

    try:
        logger.info(f"[get_bl_transit_analysis] Found {len(results)} container(s) for BL {bl_no}")
    except:
        pass

    # Parse dates
    date_cols = ['etd_lp', 'atd_lp', 'eta_dp', 'ata_dp', 'revised_eta']
    existing_date_cols = [c for c in date_cols if c in results.columns]
    if existing_date_cols:
        results = ensure_datetime(results, existing_date_cols)

    # Calculate transit metrics for each container
    container_metrics = []

    for idx, row in results.iterrows():
        metrics = {
            'container_number': row.get('container_number', 'N/A'),
            'ocean_bl_number': bl_no,
            'load_port': row.get('load_port', 'N/A'),
            'discharge_port': row.get('discharge_port', 'N/A'),
        }

        etd_lp = row.get('etd_lp')
        atd_lp = row.get('atd_lp')
        eta_dp = row.get('eta_dp')
        ata_dp = row.get('ata_dp')

        # Dates
        metrics['etd_lp'] = etd_lp.strftime('%Y-%m-%d') if pd.notna(etd_lp) else None
        metrics['atd_lp'] = atd_lp.strftime('%Y-%m-%d') if pd.notna(atd_lp) else None
        metrics['eta_dp'] = eta_dp.strftime('%Y-%m-%d') if pd.notna(eta_dp) else None
        metrics['ata_dp'] = ata_dp.strftime('%Y-%m-%d') if pd.notna(ata_dp) else None

        # Estimated transit - use pre-calculated if available
        if 'estimated_transit_days' in row.index and pd.notna(row['estimated_transit_days']):
            metrics['estimated_transit_days'] = int(row['estimated_transit_days'])
        elif pd.notna(etd_lp) and pd.notna(eta_dp):
            metrics['estimated_transit_days'] = (eta_dp - etd_lp).days
        else:
            metrics['estimated_transit_days'] = None

        # Actual transit - use pre-calculated if available
        if 'actual_transit_days' in row.index and pd.notna(row['actual_transit_days']):
            metrics['actual_transit_days'] = int(row['actual_transit_days'])
        elif pd.notna(atd_lp) and pd.notna(ata_dp):
            metrics['actual_transit_days'] = (ata_dp - atd_lp).days
        elif pd.notna(etd_lp) and pd.notna(ata_dp):
            metrics['actual_transit_days'] = (ata_dp - etd_lp).days
        else:
            metrics['actual_transit_days'] = None

        # Delay - use pre-calculated if available
        if 'delay_days' in row.index and pd.notna(row['delay_days']):
            delay = int(row['delay_days'])
            metrics['delay_days'] = delay
        elif pd.notna(eta_dp) and pd.notna(ata_dp):
            delay = (ata_dp - eta_dp).days
            metrics['delay_days'] = delay
        else:
            delay = None
            metrics['delay_days'] = None

        # Status based on delay
        if delay is not None:
            if delay > 0:
                metrics['status'] = f"Delayed by {delay} days"
            elif delay < 0:
                metrics['status'] = f"Early by {abs(delay)} days"
            else:
                metrics['status'] = "On time"
        elif pd.notna(eta_dp) and pd.isna(ata_dp):
            metrics['status'] = "In transit"
        else:
            metrics['status'] = "Unknown"

        metrics['carrier'] = row.get('final_carrier_name', 'N/A')
        metrics['transport_mode'] = row.get('transport_mode', 'N/A')
        metrics['po_numbers'] = row.get('po_number_multiple', 'N/A')

        container_metrics.append(metrics)

    # Calculate summary statistics
    transit_days = [m['actual_transit_days'] for m in container_metrics if m['actual_transit_days'] is not None]
    delay_days = [m['delay_days'] for m in container_metrics if m['delay_days'] is not None]

    summary = {
        'ocean_bl_number': bl_no,
        'total_containers': len(container_metrics),
        'arrived_containers': sum(1 for m in container_metrics if m.get('ata_dp') is not None),
        'in_transit_containers': sum(1 for m in container_metrics if m.get('ata_dp') is None),
        'avg_transit_days': round(sum(transit_days) / len(transit_days), 1) if transit_days else None,
        'avg_delay_days': round(sum(delay_days) / len(delay_days), 1) if delay_days else None,
        'delayed_containers': sum(1 for d in delay_days if d > 0),
        'on_time_or_early_containers': sum(1 for d in delay_days if d <= 0)
    }

    response = {
        'summary': summary,
        'container_details': container_metrics
    }

    try:
        logger.info(f"[get_bl_transit_analysis] Returning analysis for {len(container_metrics)} containers")
    except:
        pass

    return [response]

    return output_data

def get_bulk_container_transit_analysis(query: str) -> str:
    """
    Analyzes transit times for multiple containers or container groups.

    Supports queries like:
    - "Show transit times for all containers from SHANGHAI this month"
    - "Transit analysis for PO 6300134648"
    - "Average transit time for containers to ROTTERDAM"
    - "Transit performance by carrier MAERSK"
    - "Delayed containers in last 30 days"

    Returns aggregated transit statistics and detailed records.
    """

    query = (query or "").strip()
    query_upper = query.upper()
    query_lower = query.lower()

    # ========== 1) EXTRACT FILTERS ==========
    container_no = extract_container_number(query)
    po_no = extract_po_number(query)

    try:
        obl_no = extract_ocean_bl_number(query)
    except:
        obl_no = None

    # Extract port - **CRITICAL FIX**: Better port extraction
    load_port = None
    discharge_port = None

    # Load port extraction - improved patterns
    for pattern in [
        r'from\s+(?:load\s+port\s+)?([A-Z][A-Z\s,\-\(\)]+?)(?=\s+in\s+|\s+this\s+|\s+to\s+|\s+for\s+|$)',
        r'(?:at|from)\s+([A-Z][A-Z\s,\-\(\)]+?)(?=\s+this\s+|\s+last\s+|\s+in\s+|$)',
        r'containers?\s+from\s+([A-Z][A-Z\s,\-\(\)]+?)(?=\s+in\s+|\s+this\s+|$)'
    ]:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip().upper()
            # Exclude noise words
            if candidate and candidate not in ['THE', 'THIS', 'LAST', 'IN', 'TO', 'FOR', 'ALL', 'MONTH', 'WEEK', 'DAYS']:
                load_port = candidate
                break

    # Discharge port extraction
    match = re.search(r'to\s+(?:discharge\s+port\s+)?([A-Z\s,\-\(\)]+?)(?=\s+in\s+|\s+this\s+|$)', query, re.IGNORECASE)
    if match:
        discharge_port = match.group(1).strip().upper()

    # Extract carrier
    carrier = None
    match = re.search(r'(?:carrier|by)\s+([A-Z\s]+?)(?=\s+in\s+|$)', query, re.IGNORECASE)
    if match:
        carrier = match.group(1).strip().upper()

    # **CRITICAL FIX**: Use current date for "this month" calculations
    today = pd.Timestamp.today().normalize()

    # Extract time period with proper current month handling
    if re.search(r'\bthis\s+month\b', query_lower):
        # Use current month
        start_date = today.replace(day=1)
        # Get last day of current month
        if today.month == 12:
            end_date = today.replace(day=31)
        else:
            next_month = today.replace(month=today.month + 1, day=1)
            end_date = next_month - pd.Timedelta(days=1)
        period_desc = f"this month ({today.strftime('%B %Y')})"

        try:
            logger.info(f"[get_bulk_container_transit_analysis] Current month detected: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        except:
            pass
    elif re.search(r'\blast\s+month\b', query_lower):
        # Use previous month
        if today.month == 1:
            start_date = today.replace(year=today.year - 1, month=12, day=1)
            end_date = today.replace(year=today.year - 1, month=12, day=31)
        else:
            start_date = today.replace(month=today.month - 1, day=1)
            end_date = today.replace(day=1) - pd.Timedelta(days=1)
        period_desc = f"last month ({start_date.strftime('%B %Y')})"
    else:
        # Use parse_time_period for other cases
        start_date, end_date, period_desc = parse_time_period(query)

    # Check for specific query types
    is_delayed_query = bool(re.search(r'\bdelayed?\b', query, re.IGNORECASE))
    is_average_query = bool(re.search(r'\baverage\b|\bmean\b', query, re.IGNORECASE))

    try:
        logger.info(f"[get_bulk_container_transit_analysis] Filters: load_port={load_port}, discharge_port={discharge_port}, carrier={carrier}, period={period_desc}, dates={start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    except:
        pass

    # ========== 2) LOAD AND FILTER DATA ==========
    df = _df()

    if df.empty:
        return "No data available for your authorized consignees."

    try:
        logger.info(f"[get_bulk_container_transit_analysis] Initial dataset size: {len(df)} rows")
    except:
        pass

    # Apply filters
    mask = pd.Series(True, index=df.index)

    # Container filter
    if container_no:
        if 'container_number' not in df.columns:
            return "Container number column not found."
        clean_cont = clean_container_number(container_no)
        cont_col_norm = df["container_number"].astype(str).str.replace(r'[^A-Z0-9]', '', regex=True)
        mask &= cont_col_norm == clean_cont

        try:
            logger.info(f"[get_bulk_container_transit_analysis] After container filter: {mask.sum()} rows")
        except:
            pass

    # PO filter
    if po_no:
        po_col = 'po_number_multiple' if 'po_number_multiple' in df.columns else ('po_number' if 'po_number' in df.columns else None)
        if po_col:
            po_norm = _normalize_po_token(po_no)
            mask &= df[po_col].apply(lambda cell: _po_in_cell(cell, po_norm))

            try:
                logger.info(f"[get_bulk_container_transit_analysis] After PO filter: {mask.sum()} rows")
            except:
                pass

    # **CRITICAL FIX**: Improved port filtering with normalization
    if load_port and 'load_port' in df.columns:
        def normalize_port_for_search(port_str):
            if pd.isna(port_str):
                return ""
            s = str(port_str).upper()
            # Remove port code in parentheses
            s = re.sub(r'\([^)]*\)', '', s)
            s = s.replace(',', ' ')
            s = re.sub(r'\s+', ' ', s).strip()
            return s

        # Normalize both user input and data
        load_port_norm = normalize_port_for_search(load_port)
        df['_load_port_norm'] = df['load_port'].apply(normalize_port_for_search)

        # Try different matching strategies
        port_match = pd.Series(False, index=df.index)

        # Strategy 1: Exact match on normalized name
        port_match |= (df['_load_port_norm'] == load_port_norm)

        # Strategy 2: All words present (word boundary matching)
        if not port_match.any():
            words = [w for w in load_port_norm.split() if len(w) >= 2]
            if words:
                word_mask = pd.Series(True, index=df.index)
                for w in words:
                    word_mask &= df['_load_port_norm'].str.contains(rf'\b{re.escape(w)}\b', na=False, regex=True)
                port_match |= word_mask

        # Strategy 3: Port code match (if user provided code)
        if not port_match.any():
            # Check if load_port looks like a port code (e.g., USNYC, CNSHA)
            if re.fullmatch(r'[A-Z]{5}', load_port.replace(' ', '')):
                port_match |= df['load_port'].astype(str).str.contains(load_port, na=False, case=False)

        mask &= port_match
        df.drop(columns=['_load_port_norm'], inplace=True, errors='ignore')

        try:
            logger.info(f"[get_bulk_container_transit_analysis] After load_port filter ('{load_port}'): {mask.sum()} rows")
            if mask.sum() > 0:
                sample_ports = df.loc[mask, 'load_port'].head(5).tolist()
                logger.info(f"[get_bulk_container_transit_analysis] Sample matched load ports: {sample_ports}")
        except:
            pass

    if discharge_port and 'discharge_port' in df.columns:
        mask &= df['discharge_port'].astype(str).str.contains(discharge_port, case=False, na=False)

        try:
            logger.info(f"[get_bulk_container_transit_analysis] After discharge_port filter: {mask.sum()} rows")
        except:
            pass

    # Carrier filter
    if carrier and 'final_carrier_name' in df.columns:
        mask &= df['final_carrier_name'].astype(str).str.contains(carrier, case=False, na=False)

        try:
            logger.info(f"[get_bulk_container_transit_analysis] After carrier filter: {mask.sum()} rows")
        except:
            pass

    results = df[mask].copy()

    if results.empty:
        filters_applied = []
        if load_port:
            filters_applied.append(f"load port '{load_port}'")
        if discharge_port:
            filters_applied.append(f"discharge port '{discharge_port}'")
        if carrier:
            filters_applied.append(f"carrier '{carrier}'")
        if container_no:
            filters_applied.append(f"container '{container_no}'")
        if po_no:
            filters_applied.append(f"PO '{po_no}'")

        filter_desc = " with " + ", ".join(filters_applied) if filters_applied else ""
        return f"No containers found{filter_desc} for the {period_desc}."

    try:
        logger.info(f"[get_bulk_container_transit_analysis] After all filters: {len(results)} rows")
    except:
        pass

    # ========== 3) PARSE DATES AND CALCULATE METRICS ==========
    date_cols = ['etd_lp', 'atd_lp', 'eta_dp', 'ata_dp', 'revised_eta']
    existing_date_cols = [c for c in date_cols if c in results.columns]

    if existing_date_cols:
        results = ensure_datetime(results, existing_date_cols)

    # Calculate transit metrics
    if 'etd_lp' in results.columns and 'eta_dp' in results.columns:
        results['estimated_transit_days'] = (results['eta_dp'] - results['etd_lp']).dt.days

    if 'atd_lp' in results.columns and 'ata_dp' in results.columns:
        results['actual_transit_days'] = (results['ata_dp'] - results['atd_lp']).dt.days
    elif 'etd_lp' in results.columns and 'ata_dp' in results.columns:
        # Fallback if atd_lp not available
        results['actual_transit_days'] = (results['ata_dp'] - results['etd_lp']).dt.days

    if 'eta_dp' in results.columns and 'ata_dp' in results.columns:
        results['delay_days'] = (results['ata_dp'] - results['eta_dp']).dt.days

    # **CRITICAL FIX**: Filter by date range using actual transit completion date (ata_dp)
    if 'ata_dp' in results.columns:
        date_mask = (results['ata_dp'].notna()) & \
                   (results['ata_dp'].dt.normalize() >= start_date) & \
                   (results['ata_dp'].dt.normalize() <= end_date)
        results = results[date_mask]

        try:
            logger.info(f"[get_bulk_container_transit_analysis] After date range filter ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}): {len(results)} rows")
        except:
            pass

    # Filter delayed containers if requested
    if is_delayed_query and 'delay_days' in results.columns:
        results = results[results['delay_days'] > 0]

        try:
            logger.info(f"[get_bulk_container_transit_analysis] After delay filter: {len(results)} rows")
        except:
            pass

    if results.empty:
        return f"No containers found matching your criteria in the {period_desc}."

    # ========== 4) GENERATE STATISTICS ==========
    stats = {}

    if 'estimated_transit_days' in results.columns:
        valid_est = results['estimated_transit_days'].dropna()
        if not valid_est.empty:
            stats['avg_estimated_transit_days'] = round(valid_est.mean(), 1)
            stats['min_estimated_transit_days'] = int(valid_est.min())
            stats['max_estimated_transit_days'] = int(valid_est.max())

    if 'actual_transit_days' in results.columns:
        valid_act = results['actual_transit_days'].dropna()
        if not valid_act.empty:
            stats['avg_actual_transit_days'] = round(valid_act.mean(), 1)
            stats['min_actual_transit_days'] = int(valid_act.min())
            stats['max_actual_transit_days'] = int(valid_act.max())

    if 'delay_days' in results.columns:
        valid_delay = results['delay_days'].dropna()
        if not valid_delay.empty:
            stats['avg_delay_days'] = round(valid_delay.mean(), 1)
            stats['total_delayed'] = int((valid_delay > 0).sum())
            stats['total_on_time'] = int((valid_delay <= 0).sum())
            stats['delay_percentage'] = round((stats['total_delayed'] / len(valid_delay)) * 100, 2)

    stats['total_containers'] = len(results)
    stats['period'] = period_desc
    stats['date_range'] = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

    # Add filter context
    if load_port:
        stats['load_port_filter'] = load_port
    if discharge_port:
        stats['discharge_port_filter'] = discharge_port
    if carrier:
        stats['carrier_filter'] = carrier

    # ========== 5) PREPARE OUTPUT ==========
    if is_average_query:
        # Return just statistics
        return [stats]
    else:
        # Return statistics + detailed records
        output_cols = [
            'container_number',
            'load_port',
            'discharge_port',
            'etd_lp',
            'atd_lp',
            'ata_dp',
            'estimated_transit_days',
            'actual_transit_days',
            'delay_days',
            'final_carrier_name',
            'transport_mode'
        ]

        available_cols = [c for c in output_cols if c in results.columns]
        detail_records = results[available_cols].head(100).copy()

        # Format dates
        for dcol in ['etd_lp', 'atd_lp', 'ata_dp']:
            if dcol in detail_records.columns and pd.api.types.is_datetime64_any_dtype(detail_records[dcol]):
                detail_records[dcol] = detail_records[dcol].dt.strftime('%Y-%m-%d')

        response = {
            'summary_statistics': stats,
            'detailed_records': detail_records.to_dict(orient='records')
        }

        return [response]


# TOOLS list – must be at module level, not inside any function!
TOOLS = [

    Tool(
        name="Get Container Milestones",
        func=get_container_milestones,
        return_direct=True,
        description=(
            "PRIMARY TOOL FOR ALL CONTAINER AND PO STATUS/MILESTONE QUERIES.\n"
            "\n"
            "Use this tool FIRST for ANY query asking about:\n"
            "- Container status: 'what is status of container ABCD1234567'\n"
            "- PO status: 'what is status of PO 5300009636', 'status of the PO 5300009636'\n"
            "- Container milestones: 'show milestones for container X'\n"
            "- PO milestones: 'show milestones for PO Y', 'track PO 5300009636'\n"
            "- Container tracking: 'track container Y'\n"
            "- Container location: 'where is container Z'\n"
            "- PO location: 'where is PO 5300009636'\n"
            "- Container journey: 'event history for container'\n"
            "- PO journey: 'journey for PO X'\n"
            "\n"
            "CRITICAL PO HANDLING:\n"
            "- Input can be: 'PO 5300009636', '5300009636', 'PO5300009636', 'po 5300009636'\n"
            "- Tool will find container(s) for the PO and return their milestones\n"
            "- Searches po_number_multiple column (comma-separated values)\n"
            "- Returns: 'The Container XXXX is associated with the PO YYYY. Status is...'\n"
            "\n"
            "Keywords: status, milestone, track, tracking, where, location, journey, event, history, progress\n"
            "\n"
            "DO NOT use 'Get Containers or PO or OBL By Supplier' for status queries.\n"
            "DO NOT use 'Check Arrival Status' for milestone queries.\n"
        )
    ),

    Tool(
        name="Get Container Carrier",
        func=get_container_carrier,
        description="Get carrier information for containers or POs. For POs with multiple records, returns the carrier from the latest shipment based on ETD/ETA dates. Handles queries like 'who is the carrier for PO X', 'what carrier handles container Y', etc."
    ),
    Tool(
        name="Check Arrival Status",
        func=check_arrival_status,
        description="Check if a container or PO has arrived based on ATA_DP and derived_ATA_DP logic."
    ),
    Tool(
        name="Get Delayed Containers",
        func=get_delayed_containers,
        description=(
           "Use this tool for any question mentioning delay, late, ETA, overdue, "
           "behind schedule, missed arrival, days, or hot containers with delays. "
           "If the user mentions 'hot' and 'delay' together, use this tool."),
    ),
     Tool(
        name="Get Container ETD",
        func=get_container_etd,
        description="Return ETD for a specific container."
    ),
    Tool(
        name="Get Upcoming Arrivals",
        func=get_upcoming_arrivals,
        description=(
            "List **containers** (NOT POs) scheduled to arrive OR that have already arrived on specific dates."
            "Use ONLY when query asks about containers/shipments WITHOUT mentioning 'PO' or 'purchase order'."
            "Examples when to use:"
            "containers arriving today/tomorrow/next week"
            "containers arrived yesterday/last week"
            "show arrivals at USNYC in next 5 days"
            "containers that arrived last month"
            "DO NOT use when query mentions:"
            "'PO', 'POs', 'purchase order' → Use 'Get Upcoming POs' instead"
            "Numeric PO identifiers → Use 'Get Upcoming POs' instead"
            "Handles past/future arrivals, port filtering, date ranges."
        )
    ),
    Tool(
        name="Get Arrivals By Port",
        func=get_arrivals_by_port,
        description="Find containers arriving at a specific port or country."
    ),
	Tool(
    name="Get Containers Departing From Load Port",
    func=get_containers_departing_from_load_port,
    description=(
        "PRIMARY TOOL FOR ALL FUTURE/UPCOMING DEPARTURE QUERIES FROM LOAD PORTS.\n"
        "Use this tool for ANY query asking about containers/shipments/POs scheduled to depart or leave from origin/load ports.\n"
        "\n"
        "CRITICAL: Use this tool when query contains:\n"
        "- Future departure keywords: 'will depart', 'departing', 'scheduled to leave', 'going to leave', 'leaving', 'sailing'\n"
        "- Upcoming time references: 'next X days', 'tomorrow', 'this week', 'next week', 'upcoming', 'in next', 'within'\n"
        "- Load port references: 'from load port', 'from origin', 'from PORTNAME', 'load port PORTNAME', 'leaving PORTNAME'\n"
        "\n"
        "Examples of queries this tool handles:\n"
        "'Which containers will depart from load port QINGDAO in next 5 days?'\n"
        "'Show upcoming departures from SHANGHAI this week'\n"
        "'Containers scheduled to leave NINGBO tomorrow'\n"
        "'Which shipments are departing from BUSAN in next 7 days?'\n"
        "'POs leaving from load port HONG KONG next week'\n"
        "'What will sail from XIAMEN in next 3 days?'\n"
        "'Upcoming ETD from YANTIAN'\n"
        "'Containers going to depart from origin port QINGDAO'\n"
        "\n"
        "Technical Details:\n"
        "- Uses etd_lp (Estimated Time of Departure from Load Port)\n"
        "- Automatically excludes already departed containers (atd_lp is not null)\n"
        "- Filters by load_port column with fuzzy port name/code matching\n"
        "- Supports time windows: 'next X days', 'tomorrow', 'this/next week', date ranges\n"
        "- Returns: container_number, load_port, etd_lp, discharge_port, PO, carrier, consignee\n"
        "\n"
        "DO NOT use 'Get Containers Departed From Load Port' for future departures.\n"
        "DO NOT use 'Get Containers By ETD Window' unless query specifically asks about ETD dates without port context.\n")

   ),
	Tool(
        name="Get Containers Departed From Load Port",
        func=get_containers_departed_from_load_port,
        description="Find containers or Shipments (consider container and shipment as same)that have departed from specific load ports within a time period. Handles queries about past departures like 'containers from load port QINGDAO in last 7 days', 'containers that departed from SHANGHAI yesterday', 'which containers left NINGBO last week'. Uses atd_lp (actual departure) and etd_lp (estimated departure) with load_port filtering."
    ),
	Tool(
        name="Get Containers Missed Planned ETD",
        func=get_containers_missed_planned_etd,
        description=(
            "Use this tool when the user asks for containers that missed their planned/scheduled ETD, overdue ETD, missed departure, or did not depart as planned. "
            "Logic: atd_lp (actual departure from load port) is NULL AND etd_lp (estimated departure from load port) is before today (sysdate). "
            "Respects consignee filtering like other tools. Returns container list with etd_lp and related fields."
        )
    ),
    Tool(
        name="Keyword Lookup",
        func=lookup_keyword,
        description="Perform a fuzzy keyword search across all shipment data fields."
    ),
    Tool(
        name="Analyze Data with Pandas",
        func=analyze_data_with_pandas,
        description="Analyze shipping data using pandas based on a natural language query."
    ),
    Tool(
        name="Get Field Info",
        func=get_field_info,
        description="Retrieve detailed information for a container or summarize a specific field."
    ),
    Tool(
        name="Get Vessel Info",
        func=get_vessel_info,
        description="Get vessel details for a specific container."
    ),
    Tool(
        name="Get Upcoming POs",
        func=get_upcoming_pos,
        description=(
            "PRIMARY TOOL FOR ALL PO/PURCHASE ORDER ARRIVAL QUERIES"
            "CRITICAL: Use this tool if query contains ANY of these keywords:"
            "- 'PO', 'pos', 'purchase order', 'P.O.', 'po number'"
            "- Numeric patterns like '5302816722', '6300134648'"
            "- Phrases: 'which POs', 'POs arriving', 'POs scheduled', 'upcoming POs'"
            "Handles ALL PO-related arrival queries:"
            "which POs are arriving in next 5 days?"
            "POs arriving at DEHAM in 7 days"
            "show upcoming POs for consignee 0000866"
            "'list POs scheduled for USNYC next week'\n"
            "is PO 5302816722 arriving this week?"
            "ALWAYS use this tool when:"
            "1. Query explicitly mentions 'PO' or 'purchase order'"
            "2. Query asks about 'POs' (plural)\n"
            "3. Query contains numeric PO identifiers (e.g., 5302816722)"
            "DO NOT use 'Get Upcoming Arrivals' for PO queries."
            "Supports: port/location filtering, consignee filtering, time periods, hot flag."
        )
    ),
    Tool(
        name="Get Delayed POs",
        func=get_delayed_pos,
        description="Find PO's that are delayed based on delivery/empty-container status and ETA logic."
    ),
    #Tool(
    #    name="Get Containers Arriving Soon",
    #    func=get_containers_arriving_soon,
    #    description="List containers arriving soon (ETA window, ATA is null)."
    #),
    Tool(
        name="Get Top Values For Column",
        func=get_top_values_for_column,
        description="Get the top 5 most frequent values for a specified column."
    ),
    Tool(
        name="Get Load Port For Container",
        func=get_load_port_for_container,
        description="Get the load port details for a specific container."
    ),
    Tool(
        name="Answer With Column Mapping",
        func=answer_with_column_mapping,
        description="Interprets user queries using robust synonym mapping and answers using the correct column(s)."
    ),
    Tool(
        name="Vector Search",
        func=vector_search_tool,
        description="Search the vector database for relevant shipment information using semantic similarity."
    ),
    Tool(
        name="SQL Query Tool",
        func=sql_query_tool,
        description="Execute natural language queries against the shipment database. Ask questions like 'Show me delayed containers', 'How many shipments from Singapore?', etc. No SQL knowledge required - just ask in plain English!"
    ),
    Tool(
        name="Check Transit Status",
        func=check_transit_status,
        description="Check if a PO/cargo is currently in transit or find containers with transit times exceeding specific thresholds. Handles questions like 'which containers are taking more than X days of transit time?'"
    ),
    Tool(
        name="Get Containers By Carrier",
        func=get_containers_by_carrier,
        description="Get containers handled or shipped by a specific carrier in recent days"
    ),
    Tool(
        name="Get Supplier In Transit",
        func=get_supplier_in_transit,
        description="Containers/POs from a supplier that are still in transit (ata_dp null, not delivered, empty return null)."
    ),
    Tool(
        name="Get Supplier Last Days",
        func=get_supplier_last_days,
        description="Containers from a supplier that have arrived in the last N days (ata_dp within window)."
    ),
    Tool(
        name="Get Containers or PO or OBL By Supplier",
        func=get_containers_PO_OBL_by_supplier,
        description="use this function when user query mentions supplier or shipper name to get related containers or POs or OBLs.dont look for any other function if user query mentions supplier or shipper name."
    ),
    Tool(
        name="Check PO Month Arrival",
        func=check_po_month_arrival,
        description="Check if a PO can arrive by the end of current month"
    ),
    Tool(
        name="Get Weekly Status Changes",
        func=get_weekly_status_changes,
        description="Get container status changes for current or last week"
    ),
    Tool(
        name="Get Hot Upcoming Arrivals",
        func=get_hot_upcoming_arrivals,
        description="List hot containers (and related POs) arriving within next N days."
    ),
    Tool(
    name="Get Hot Containers",
    func=get_hot_containers,
    description=(
        "PRIMARY TOOL FOR ALL HOT/PRIORITY CONTAINER QUERIES.\n"
        "\n"
        "Use this tool for ANY query that mentions 'hot', 'priority', 'urgent', 'rush', or 'expedited' containers.\n"
        "\n"
        "CRITICAL: This tool handles BOTH:\n"
        "1. Generic hot container queries: 'Show my hot containers', 'List all priority shipments'\n"
        "2. Hot containers WITH filters:\n"
        "   - Delay filters: 'hot containers delayed by more than 3 days', 'priority shipments late by 5 days'\n"
        "   - Port filters: 'hot containers at Rotterdam', 'urgent shipments arriving at NLRTM'\n"
        "   - Transport mode: 'hot containers by sea', 'priority shipments by air'\n"
        "   - Arrival status: 'hot containers that have arrived', 'priority shipments already reached'\n"
        "\n"
        "Examples of queries this tool handles:\n"
        "- 'Show me all hot containers'\n"
        "- 'List priority shipments'\n"
        "- 'Hot containers delayed by more than 3 days'\n"
        "- 'Which hot containers are late by 5 days?'\n"
        "- 'Urgent shipments at USNYC'\n"
        "- 'Priority containers by sea'\n"
        "- 'Hot containers that have arrived'\n"
        "\n"
        "DO NOT use 'Get Delayed Containers' if query mentions 'hot' or 'priority'.\n"
        "DO NOT use 'Get Hot Upcoming Arrivals' unless query explicitly asks about upcoming/arriving/next X days.\n"
        "\n"
        "This tool automatically:\n"
        "- Filters to ONLY hot_container_flag = TRUE\n"
        "- Applies delay calculations when delay keywords present\n"
        "- Filters by port when location mentioned\n"
        "- Filters by transport mode when sea/air/road mentioned\n"
        "- Returns all relevant hot containers without date restrictions (unless delay filter applied)"
    )
),
    Tool(
        name="Get Delayed BLs",
        func=get_delayed_bls,
        description="Find ocean BLs (ocean_bl_no_multiple) that are delayed. Supports BL tokens in query, consignee filter, location filters and numeric delay filters (e.g., 'delayed by 5 days')."
    ),
    Tool(
        name="Get Carrier For PO",
        func=get_carrier_for_po,
        description="Find the final_carrier_name for a PO (matches po_number_multiple / po_number). Use queries like 'who is carrier for PO 5500009022' or '5500009022'."
    ),
    Tool(
        name="Is PO Hot",
        func=is_po_hot,
        description="Check whether a PO is marked hot via the container's hot flag (searches po_number_multiple / po_number)."
    ),
    Tool(
        name="Extract Transport Modes",
        func=extract_transport_modes,
        description="Parse transport mode tokens from a user query and return normalized set e.g. 'sea', 'air', 'road', 'rail', 'courier', 'sea-air'."
    ),
    Tool(
        name="Get Containers By Transport Mode",
        func=get_containers_by_transport_mode,
        description="Find containers filtered by transport_mode (e.g. 'arrived by sea', 'arrive by air in next 3 days')."
    ),
    Tool(
        name="Find Ocean BL Column",
        func=_find_ocean_bl_col,
        description="Identify the best-matching column name for the ocean BL field in the dataset (e.g. ocean_bl_no_multiple or variants)."
    ),

    Tool(
        name="Get Containers For BL",
        func=get_containers_for_bl,
        description="Find container(s) and basic status for an ocean BL (matches ocean_bl_no_multiple). Use queries like 'is MOLWMNL2400017 reached to discharge port?' or 'which container has bill of lading MOLWMNL2400017?'."
    ),
    Tool(
        name="Get Carrier For BL",
        func=get_carrier_for_bl,
        description="Return the final_carrier_name for an ocean BL value (matches ocean_bl_no_multiple). Use queries like 'who is carrier for BL MOLWMNL2400017' or 'MOLWMNL2400017'."
    ),
    Tool(
        name="Is BL Hot",
        func=is_bl_hot,
        description="Check whether an ocean BL is marked hot via its container's hot flag (searches ocean_bl_no_multiple)."
    ),
    Tool(
        name="Handle Non-shipping queries",
        func=handle_non_shipping_queries,
        description="This is for non-shipping generic queries. Like 'how are you' or 'hello' or 'hey' or 'who are you' etc."
    ),
    Tool(
        name="Get Containers By Final Destination",
        func=get_containers_by_final_destination,
        description="Find containers arriving at a specific final destination/distribution center (FD/DC) within a timeframe. Handles queries like 'containers arriving at FD Nashville in next 3 days' or 'list containers to DC Phoenix next week'."
    ),
	Tool(
        name="Get Upcoming BLs",
        func=get_upcoming_bls,
        description="List upcoming ocean BLs. Handles queries with transport mode, location, and timeframes, like 'Show me BLs arriving by sea at NLRTM in next 10 days'."
    ),
    Tool(
        name="Get ETA For PO",
        func=get_eta_for_po,
        description="Get ETA for a PO (prefers revised_eta over eta_dp)."
    ),
	Tool(
        name="Get Containers By ETD Window",
        func=get_containers_by_etd_window,
        description="List containers whose ETD (etd_lp) falls within a time window parsed from the query (e.g., 'Which containers have ETD in the next 7 days?'). Supports consignee filtering."
    ),
    Tool(
    name="Get Container Transit Analysis",
    func=get_container_transit_analysis,
    description="""Analyzes transit times, delays, and journey metrics for a specific container.
    Use for queries about:
    - Estimated transit time for a container
    - Actual transit time for a container
    - Container delays and delivery status
    - Journey analysis with route details
    - Transit efficiency metrics

    Supports additional filters:
    - Load/discharge port: "transit time for ABCD1234567 from Shanghai to Rotterdam"
    - Carrier: "delay for WXYZ9876543 by Maersk"
    - Transport mode: "transit analysis for TCLU4521258 by sea"
    - Date range: "journey time for HASU5127798 in November"

    Input: Query with container number (filters optional)
    Output: Detailed transit metrics including dates, transit days, delays, efficiency"""
    ),
    Tool(
    name="Get PO Transit Analysis",
    func=get_po_transit_analysis,
    description="""Analyzes transit times for ALL containers in a specific PO.
    Use for queries about:
    - "Transit analysis for PO 6300134648"
    - "Show transit times for purchase order 5302816722"
    - "What is the transit performance for PO 6300134648?"
    - "Delay analysis for PO 5302816722"
    - "How are containers for PO 6300134648 performing?"

    Returns:
    - Summary statistics (avg transit time, avg delay, container counts)
    - Individual transit metrics for each container in the PO
    - Arrived vs in-transit container breakdown

    Input: Query with PO number
    Output: Summary + detailed metrics for all containers"""
    ),
    Tool(
    name="Get BL Transit Analysis",
    func=get_bl_transit_analysis,
    description="""Analyzes transit times for ALL containers in a specific Ocean BL.
    Use for queries about:
    - "Transit analysis for BL MOLWMNL2400017"
    - "Show transit times for ocean BL MOLWMNL2400017"
    - "What is the transit performance for bill of lading MOLWMNL2400017?"
    - "Delay analysis for BL MOLWMNL2400017"

    Returns:
    - Summary statistics (avg transit time, avg delay, container counts)
    - Individual transit metrics for each container in the BL
    - Arrived vs in-transit container breakdown

    Input: Query with ocean BL number
    Output: Summary + detailed metrics for all containers"""
    ),
    Tool(
        name="Get ETA For Booking",
        func=get_eta_for_booking,
        description=(
            "PRIMARY TOOL for BOOKING NUMBER ETA/arrival queries.\n"
            "Use this tool when the user asks about ETA/arrival/status for a booking number like:\n"
            "- 'what is the ETA of Booking GT3000512?'\n"
            "- 'ETA for booking CN9140225'\n"
            "- 'has booking <BOOKING> arrived?'\n"
            "\n"
            "Data source: booking_number_multiple (comma-separated booking numbers).\n"
            "ETA priority: revised_eta (if present) else eta_dp.\n"
            "Arrival logic: arrived if ata_dp is present; else use derived_ata_dp logic (<= today => arrived, otherwise expected).\n"
            "\n"
            "DO NOT use Keyword Lookup / Container Milestones for booking ETA questions.\n"
        )
    ),
    Tool(
        name="Get Booking Details",
        func=get_booking_details,
        description=(
            "PRIMARY TOOL for BOOKING NUMBER lookups (booking ⇄ PO ⇄ container ⇄ ocean BL/OBL).\n"
            "\n"
            "USE THIS TOOL when user asks about BOOKING NUMBERS:\n"
            "CORRECT usage examples:\n"
            "- 'what is the booking number of PO 5300008696' → Input: '5300008696'\n"
            "- 'booking number of 5300008696' → Input: '5300008696'\n"
            "- 'booking number for PO 5300008696' → Input: '5300008696'\n"
            "- 'what is the booking number of container MSBU4522691' → Input: 'MSBU4522691'\n"
            "- 'booking number of MSBU4522691' → Input: 'MSBU4522691'\n"
            "- 'booking number of BL MOLWMNL2400017' → Input: 'MOLWMNL2400017'\n"
            "- 'PO of booking VN2084805' → Input: 'VN2084805'\n"
            "- 'container of booking TH2017962' → Input: 'TH2017962'\n"
            "- 'BL of booking GT3000512' → Input: 'GT3000512'\n"
            "\n"
            "PO NUMBER HANDLING (CRITICAL):\n"
            "- PO numbers are typically 7-10 digits (e.g., 5300008696, 5300009636)\n"
            "- Input can be: 'PO 5300008696', '5300008696', 'PO5300008696', 'po 5300008696'\n"
            "- Tool searches po_number_multiple column (comma-separated values)\n"
            "- Returns booking_number_multiple + container + discharge_port + consignee\n"
            "\n"
            " CONTAINER HANDLING:\n"
            "- Container format: 4 letters + 7 digits (e.g., MSBU4522691, MRKU0496086)\n"
            "- Returns booking_number_multiple for that container\n"
            "\n"
            " BARE IDENTIFIER SUPPORT:\n"
            "Agent retries may pass only the code without context words.\n"
            "This tool MUST handle:\n"
            "- '5300008696' (7+ digits) → Treat as PO number → return booking\n"
            "- 'MSBU4522691' (4 letters + 7 digits) → Treat as container → return booking\n"
            "- 'MOLWMNL2400017' (ocean BL pattern) → Treat as BL → return booking\n"
            "- 'VN2084805' (booking pattern) → Return PO/container/BL mapping\n"
            "\n"
            " DO NOT USE for:\n"
            "- Status queries (use 'Get Container Milestones')\n"
            "- Milestone queries (use 'Get Container Milestones')\n"
            "- ETA of PO (use 'Get ETA For PO')\n"
            "- Transit analysis (use transit tools)\n"
            "\n"
            " ETA delegation: 'ETA of booking X' → delegates to 'Get ETA For Booking'\n"
        )
    ),
	Tool(
        name="Get Containers Still At Load Port (Not Yet Departed)",
        func=get_containers_still_at_load_port,
        description=(
            "PRIMARY TOOL for queries asking which containers/shipments are STILL AT an origin/load port and have NOT departed yet.\n"
            "\n"
            "Use this tool when user asks variants like:\n"
            "- 'Which containers are still at CHATTOGRAM and not yet departed?'\n"
            "- 'containers not departed from Chattogram'\n"
            "- 'shipments stuck at origin Chattogram'\n"
            "- 'containers awaiting departure at BDCGP'\n"
            "- 'how many containers are at SHANGHAI and haven't left?'\n"
            "\n"
            "Core logic (must follow):\n"
            "- atd_lp is NULL => not departed\n"
            "- Filter by load_port matching the user location (port name like 'CHATTOGRAM' OR code like '(BDCGP)' or 'BDCGP')\n"
            "\n"
            "Supported optional filters (only if user asks):\n"
            "- Hot containers: if query includes hot/priority/urgent/rush/expedited (uses hot_container_flag column)\n"
            "- Transport mode: sea/air/road/rail/courier/sea-air\n"
            "- Consignee scope: honors thread-local consignee authorization via _df(); can also accept consignee_code parameter.\n"
            "\n"
            "Important:\n"
            "- DO NOT use 'Get Containers Departing From Load Port' (that tool is ETD-window/upcoming departures).\n"
            "- This tool is for 'still at X', 'not yet departed', 'awaiting departure', 'not left' semantics.\n"
        ),
    ),
    Tool(
    name="Get Bulk Container Transit Analysis",
    func=get_bulk_container_transit_analysis,
    description="""Analyzes transit times for multiple containers with filtering options.
    Use for queries about:
    - Average transit times by port/carrier
    - Delayed containers analysis
    - Transit performance statistics
    - Bulk container metrics
    Input: Query with filters (port, carrier, date range, etc.)"""
    ),

]

