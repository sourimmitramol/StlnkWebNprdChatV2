# agents/tools.py
import logging
#from utils.misc import ensure_datetime
import re
from datetime import datetime, timedelta
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.chat_models import AzureChatOpenAI
from config import settings
import pandas as pd
from fuzzywuzzy import process
from langchain.agents import Tool
from services.vectorstore import get_vectorstore
from utils.container import extract_container_number,extract_po_number,extract_ocean_bl_number
from utils.logger import logger
from services.azure_blob import get_shipment_df
from utils.misc import to_datetime, clean_container_number
from agents.prompts import map_synonym_to_column
from agents.prompts import COLUMN_SYNONYMS
from services.vectorstore import get_vectorstore
from langchain_openai import AzureChatOpenAI
import sqlite3
from sqlalchemy import create_engine
import threading
from difflib import get_close_matches





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

# Then replace all sort_values calls in your functions:
# OLD: df.sort_values(column, na_last=True)
# NEW: df = safe_sort_dataframe(df, column)
# ------------------------------------------------------------------
# Helper – give every tool a clean copy of the DataFrame
# ------------------------------------------------------------------
def _df() -> pd.DataFrame:
    """
    Get filtered DataFrame based on thread-local consignee codes.
    Returns the full dataset if no consignee codes are set.
    """
    from services.azure_blob import get_shipment_df
    df = get_shipment_df()
    
    # Check if consignee codes are set in thread-local storage
    import threading
    if hasattr(threading.current_thread(), 'consignee_codes'):
        consignee_codes = threading.current_thread().consignee_codes
        
        if consignee_codes and 'consignee_code_multiple' in df.columns:
            import re
            # Extract numeric codes from consignee codes
            numeric_codes = []
            for code in consignee_codes:
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


def handle_non_shipping_queries(query: str) -> dict:
    """
    Handle greetings, thanks, small talk, and other non-shipping queries.
    Always returns a friendly Anna response.
    """

    q = query.lower().strip()

    # Greetings
    greetings = ["hi", "hello", "hey", "gm", "good morning", "good afternoon", "good evening", "hola"]
    if any(word in q for word in greetings):
        return "Hello! I’m MCS AI, your shipping assistant. How can I help you today?"

    # Thanks
    thanks = ["thank", "thx", "thanks", "thank you", "ty", "much appreciated"]
    if any(word in q for word in thanks):
        return "You’re very welcome! Always happy to help. – MCS AI"

    # How are you / small talk
    if "how are you" in q or "how r u" in q:
        return "I’m doing great, thanks for asking! How about you? – MCS AI"

    # Who are you / introduction
    if "who are you" in q or "your name" in q or "what is your name" in q:
        return "I’m MCS AI, your AI-powered shipping assistant. I can help you track containers, POs, and more."

    # Goodbye
    farewells = ["bye", "goodbye", "see you", "take care", "cya", "see ya"]
    if any(word in q for word in farewells):
        return "Goodbye! Have a wonderful day ahead. – MCS AI"

    # Fallback for anything non-shipping
    return "That doesn’t look like a shipping-related question, but I’m MCS AI and I’m here to help! 😊 What would you like to know?"




def _get_current_consignee_codes():
    """Get current consignee codes from thread-local storage"""
    import threading
    return getattr(threading.current_thread(), 'consignee_codes', None)

def _df_filtered_by_consignee(consignee_codes=None):
    """Get DataFrame filtered by consignee codes if provided"""
    df = get_shipment_df()
    if consignee_codes:
        import re
        # Extract just the numeric part from consignee codes
        numeric_codes = []
        for code in consignee_codes:
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

# ...existing code...

def get_hot_upcoming_arrivals(query: str) -> str:
    """
    List hot containers (and related POs) arriving within next N days.
    - Per-row ETA selection: use revised_eta if present, otherwise eta_dp.
    - Exclude rows where ata_dp is NOT null (already arrived).
    - Default days = 7. Query examples: "next 3 days", "in next 5 days".
    Returns list[dict] (container_number, po_number_multiple, discharge_port, revised_eta, eta_dp).
    """
    # parse days
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
 
    today = pd.Timestamp.today().normalize()
    end_date = today + pd.Timedelta(days=n_days)
 
    # <-- ADDED: log parsed timeframe for debugging
    try:
        logger.info(f"[get_arrivals_by_port] query={query!r} parsed_n_days={n_days} today={today.strftime('%Y-%m-%d')} end_date={end_date.strftime('%Y-%m-%d')}")
    except Exception:
        print(f"[get_arrivals_by_port] parsed_n_days={n_days} today={today} end_date={end_date}")
    df = _df()  # respects consignee filtering
 
    # transport mode filter (if mentioned in query)
    modes = extract_transport_modes(query)
    if modes and 'transport_mode' in df.columns:
        df = df[df['transport_mode'].astype(str).str.lower().apply(lambda s: any(m in s for m in modes))]
 
    # find hot flag column
    hot_flag_cols = [c for c in df.columns if 'hot_container_flag' in c.lower()]
    if not hot_flag_cols:
        hot_flag_cols = [c for c in df.columns if 'hot_container' in c.lower()]
    if not hot_flag_cols:
        return "No hot-container flag column found in the dataset."
 
    hot_col = hot_flag_cols[0]
 
    # select hot rows
    hot_mask = df[hot_col].astype(str).str.strip().str.upper().isin({'Y', 'YES', 'TRUE', '1', 'HOT'})
    hot_df = df[hot_mask].copy()
    if hot_df.empty:
        return "No hot containers found for your authorized consignees."
 
    # determine per-row ETA using revised_eta then eta_dp
    date_priority = [c for c in ['revised_eta', 'eta_dp'] if c in hot_df.columns]
    if not date_priority:
        return "No ETA columns (revised_eta / eta_dp) found in the data to compute upcoming arrivals."
 
    parse_cols = date_priority.copy()
    if 'ata_dp' in hot_df.columns:
        parse_cols.append('ata_dp')
    hot_df = ensure_datetime(hot_df, parse_cols)
 
    if 'revised_eta' in hot_df.columns and 'eta_dp' in hot_df.columns:
        hot_df['eta_for_filter'] = hot_df['revised_eta'].where(hot_df['revised_eta'].notna(), hot_df['eta_dp'])
    elif 'revised_eta' in hot_df.columns:
        hot_df['eta_for_filter'] = hot_df['revised_eta']
    else:
        hot_df['eta_for_filter'] = hot_df['eta_dp']
 
    # filter: eta_for_filter between today..end_date and ata_dp is null (not arrived)
    date_mask = (hot_df['eta_for_filter'] >= today) & (hot_df['eta_for_filter'] <= end_date)
    if 'ata_dp' in hot_df.columns:
        date_mask &= hot_df['ata_dp'].isna()
 
    result = hot_df[date_mask].copy()
    if result.empty:
        return f"No hot containers (or related POs) arriving between {today.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}."
 
    # prepare output columns and format dates
    out_cols = ['container_number', 'po_number_multiple', 'discharge_port', 'revised_eta', 'eta_dp', 'eta_for_filter']
    out_cols = [c for c in out_cols if c in result.columns]
    out_df = result[out_cols].sort_values('eta_for_filter').head(50).copy()
 
    for d in ['revised_eta', 'eta_dp', 'eta_for_filter']:
        if d in out_df.columns and pd.api.types.is_datetime64_any_dtype(out_df[d]):
            out_df[d] = out_df[d].dt.strftime('%Y-%m-%d')
 
    if 'eta_for_filter' in out_df.columns:
        out_df = out_df.drop(columns=['eta_for_filter'])
 
    return out_df.where(pd.notnull(out_df), None).to_dict(orient='records')




# ...existing code...
# (query: str) -> str:
#     """
#     Get list of hot containers for specified consignee codes.
#     Also supports:
#       1) which hot container are delayed/late at <PORT>
#       2) which hot containers are delayed/late by X days at <PORT>
#       3) which hot containers will arrive at <PORT>
#       4) which hot containers will arrive at <PORT> in next X days

#     - Strict port code filter: include only rows whose discharge_port OR vehicle_arrival_lcn contains '(CODE)'
#     - Delay semantics:
#         > N → delay_days > N
#         >= N / at least N / N or more → delay_days >= N
#         up to N / within N / max N / no more than N → 0 < delay_days ≤ N
#         exactly N / plain 'N days' → delay_days == N
#         default (no number) → delay_days > 0
#     - Arrivals semantics:
#         upcoming window defaults to 7 days when not specified; excludes already-arrived (ata_dp is null)
#     """
#     import re

#     df = _df()  # This automatically filters by consignee


#     # Find hot flag column
#     hot_flag_cols = [c for c in df.columns if 'hot_container_flag' in c.lower()] or \
#                     [c for c in df.columns if 'hot_container' in c.lower()]
#     if not hot_flag_cols:
#         return "Hot container flag column not found in the data."
#     hot_flag_col = hot_flag_cols[0]

#     # Robust hot detector
#     def _is_hot(v) -> bool:
#         if pd.isna(v):
#             return False
#         s = str(v).strip().upper()
#         return s in {"Y", "YES", "TRUE", "1", "HOT"} or v is True or v == 1

#     # Filter to hot rows
#     hot_mask = df[hot_flag_col].apply(_is_hot)
#     hot_df = df[hot_mask].copy()
#     if hot_df.empty:
#         return "No hot containers found for your authorized consignees."

#     # Optional strict location filter (code or name)
#     port_cols = [c for c in ["discharge_port", "vehicle_arrival_lcn", "final_destination", "place_of_delivery"] if c in hot_df.columns]

#     def _extract_loc_code_and_name(q: str):
#         q_up = (q or "").upper()
#         # Prefer explicit code in parentheses, e.g., (NLRTM)
#         m = re.search(r"\(([A-Z0-9]{3,6})\)", q_up)
#         if m:
#             return m.group(1), None
#         # Bare code tokens (3–6 alnum) only if present in dataset codes
#         cand_codes = set(re.findall(r"\b[A-Z0-9]{3,6}\b", q_up))
#         if port_cols and cand_codes:
#             known_codes = set()
#             for c in port_cols:
#                 vals = hot_df[c].dropna().astype(str).str.upper()
#                 known_codes |= set(re.findall(r"\(([A-Z0-9]{3,6})\)", " ".join(vals.tolist())))
#             for code in cand_codes:
#                 if code in known_codes:
#                     return code, None
#         # Fallback name after at|in|to
#         m2 = re.search(r"(?:\bat|\bin|\bto)\s+([A-Z][A-Z\s\.,'-]{3,})$", q_up)
#         name = m2.group(1).strip() if m2 else None
#         return None, name

#     code, name = _extract_loc_code_and_name(query)
#     if code or name:
#         loc_mask = pd.Series(False, index=hot_df.index)
#         if code:
#             pat = rf"\({re.escape(code)}\)"
#             for c in port_cols:
#                 loc_mask |= hot_df[c].astype(str).str.upper().str.contains(pat, na=False)
#         else:
#             tokens = [w for w in re.split(r"\W+", name or "") if len(w) >= 3]
#             for c in port_cols:
#                 col_vals = hot_df[c].astype(str).str.upper()
#                 cond = pd.Series(True, index=hot_df.index)
#                 for t in tokens:
#                     cond &= col_vals.str.contains(re.escape(t), na=False)
#                 loc_mask |= cond
#         hot_df = hot_df[loc_mask].copy()
#         if hot_df.empty:
#             where = f"{code or name}"
#             return f"No hot containers found at {where} for your authorized consignees."

#     ql = (query or "").lower()

#     # Branch A: delayed/late hot containers (arrived only)
#     if any(w in ql for w in ("delay", "late", "overdue", "behind")):
#         hot_df = ensure_datetime(hot_df, ["eta_dp", "ata_dp"])
#         arrived = hot_df[hot_df["ata_dp"].notna()].copy()
#         if arrived.empty:
#             where = f" at {code or name}" if (code or name) else ""
#             return f"No hot containers have arrived{where} for your authorized consignees."

#         arrived["delay_days"] = (arrived["ata_dp"] - arrived["eta_dp"]).dt.days
#         arrived["delay_days"] = arrived["delay_days"].fillna(0).astype(int)

#         # Parse numeric qualifiers
#         more_than   = re.search(r"(?:more than|over|>\s*)\s*(\d+)\s*days?", ql)
#         at_least    = re.search(r"(?:at\s+least|>=\s*|or\s+more|minimum)\s*(\d+)\s*days?", ql)
#         up_to       = re.search(r"(?:up\s*to|no\s*more\s*than|within|maximum)\s*(\d+)\s*days?", ql)
#         exact       = re.search(r"(?:delayed|late|overdue|behind)?\s*by?\s*(\d+)\s*days?", ql) or re.search(r"\b(\d+)\s*days?\b", ql)

#         if more_than:
#             d = int(more_than.group(1)); delayed = arrived[arrived["delay_days"] > d]
#         elif at_least:
#             d = int(at_least.group(1)); delayed = arrived[arrived["delay_days"] >= d]
#         elif up_to:
#             d = int(up_to.group(1)); delayed = arrived[(arrived["delay_days"] > 0) & (arrived["delay_days"] <= d)]
#         elif exact:
#             d = int(exact.group(1)); delayed = arrived[arrived["delay_days"] == d]
#         else:
#             delayed = arrived[arrived["delay_days"] > 0]

#         # Belt-and-suspenders: ensure they are still HOT
#         if hot_flag_col in delayed.columns:
#             delayed = delayed[delayed[hot_flag_col].apply(_is_hot)]
#         delayed = delayed[delayed["delay_days"] > 0]

#         if delayed.empty:
#             where = f" at {code or name}" if (code or name) else ""
#             return f"No hot containers are delayed for your authorized consignees{where}."

#         cols = ["container_number", "eta_dp", "ata_dp", "delay_days", "discharge_port"]
#         if "vehicle_arrival_lcn" in delayed.columns:
#             cols.append("vehicle_arrival_lcn")
#         cols = [c for c in cols if c in delayed.columns]
#         out = delayed[cols].sort_values("delay_days", ascending=False).head(100).copy()

#         # Format dates
#         for dcol in ["eta_dp", "ata_dp"]:
#             if dcol in out.columns and pd.api.types.is_datetime64_any_dtype(out[dcol]):
#                 out[dcol] = out[dcol].dt.strftime("%Y-%m-%d")

#         return out.where(pd.notnull(out), None).to_dict(orient="records")

#     # Branch B: upcoming arrivals of hot containers (ATA null)
#     if any(w in ql for w in ("arriv", "expected", "due", "will arrive", "arriving soon", "next")):
#         # Parse "next X days" (default 7 if not specified)
#         days = None
#         for pat in [
#             r"(?:next|upcoming|within|in)\s+(\d{1,3})\s+days?",
#             r"arriving.*?(\d{1,3})\s+days?",
#             r"will.*?arrive.*?(\d{1,3})\s+days?",
#             r"(\d{1,3})\s+days?",
#         ]:
#             m = re.search(pat, query, re.IGNORECASE)
#             if m:
#                 days = int(m.group(1)); break
#         n_days = days if days is not None else 7

#         # ETA preference per-row and exclude arrived
#         date_priority = [c for c in ['revised_eta', 'eta_dp'] if c in hot_df.columns]
#         if not date_priority:
#             return "No ETA columns (revised_eta / eta_dp) found in the data."
#         parse_cols = date_priority.copy()
#         if 'ata_dp' in hot_df.columns:
#             parse_cols.append('ata_dp')
#         hot_df = ensure_datetime(hot_df, parse_cols)

#         if 'revised_eta' in hot_df.columns and 'eta_dp' in hot_df.columns:
#             hot_df['eta_for_filter'] = hot_df['revised_eta'].where(hot_df['revised_eta'].notna(), hot_df['eta_dp'])
#         elif 'revised_eta' in hot_df.columns:
#             hot_df['eta_for_filter'] = hot_df['revised_eta']
#         else:
#             hot_df['eta_for_filter'] = hot_df['eta_dp']

#         today = pd.Timestamp.today().normalize()
#         end_date = today + pd.Timedelta(days=n_days)
#         mask = hot_df['eta_for_filter'].notna() & (hot_df['eta_for_filter'] >= today) & (hot_df['eta_for_filter'] <= end_date)
#         if 'ata_dp' in hot_df.columns:
#             mask &= hot_df['ata_dp'].isna()
#         upcoming = hot_df[mask].copy()

#         # Ensure still HOT (in case of column drops/joins)
#         if hot_flag_col in upcoming.columns:
#             upcoming = upcoming[upcoming[hot_flag_col].apply(_is_hot)]

#         if upcoming.empty:
#             where = f" at {code or name}" if (code or name) else ""
#             return f"No hot containers arriving between {today.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}{where}."

#         cols = ['container_number', 'discharge_port', 'revised_eta', 'eta_dp', 'eta_for_filter','delay_days']
#         if 'vehicle_arrival_lcn' in upcoming.columns:
#             cols.append('vehicle_arrival_lcn')
#         cols = [c for c in cols if c in upcoming.columns]
#         out = upcoming[cols].sort_values('eta_for_filter').head(100).copy()

#         for d in ['revised_eta', 'eta_dp', 'eta_for_filter']:
#             if d in out.columns and pd.api.types.is_datetime64_any_dtype(out[d]):
#                 out[d] = out[d].dt.strftime('%Y-%m-%d')

#         if 'eta_for_filter' in out.columns:
#             out = out.rename(columns={'eta_for_filter': 'eta'})

#         return out.where(pd.notnull(out), None).to_dict(orient='records')

#     # Fallback: simple hot listing (no delay/arrival/port qualifiers)
#     display_cols = ['container_number', 'consignee_code_multiple', 'delay_days']
#     display_cols += [c for c in ['discharge_port', 'eta_dp', 'ata_dp'] if c in hot_df.columns]
#     display_cols = [c for c in display_cols if c in hot_df.columns]

#     if 'eta_dp' in hot_df.columns:
#         hot_df = safe_sort_dataframe(hot_df, 'eta_dp', ascending=True)
#     else:
#         hot_df = safe_sort_dataframe(hot_df, 'container_number', ascending=True)

#     result_data = hot_df[display_cols].head(200).copy()
#     for col in result_data.columns:
#         if pd.api.types.is_datetime64_dtype(result_data[col]):
#             result_data[col] = result_data[col].dt.strftime('%Y-%m-%d')

#     if len(result_data) == 0:
#         return "No hot containers found for your authorized consignees."
#     return result_data.where(pd.notnull(result_data), None).to_dict(orient="records")
# # ...existing code...



# def get_hot_containers_by_consignee(query: str) -> str:
#     """
#     Get hot containers filtered by specific consignee codes mentioned in the query.
#     Input: Query mentioning hot containers and consignee codes.
#     Output: Hot containers for specified consignees.
#     """
#     # Extract consignee codes from query if mentioned
#     consignee_pattern = r'consignee[s]?\s+(?:code[s]?\s+)?([0-9,\s]+)'
#     consignee_match = re.search(consignee_pattern, query, re.IGNORECASE)
    
#     if consignee_match:
#         # Parse consignee codes from the query
#         codes_str = consignee_match.group(1)
#         extracted_codes = [code.strip() for code in re.split(r'[,\s]+', codes_str) if code.strip().isdigit()]
        
#         if extracted_codes:
#             # Temporarily set these codes in thread context
#             import threading
#             original_codes = getattr(threading.current_thread(), 'consignee_codes', None)
#             threading.current_thread().consignee_codes = extracted_codes
            
#             try:
#                 result = get_hot_containers(query)
#                 return result
#             finally:
#                 # Restore original codes
#                 if original_codes:
#                     threading.current_thread().consignee_codes = original_codes
#                 else:
#                     if hasattr(threading.current_thread(), 'consignee_codes'):
#                         delattr(threading.current_thread(), 'consignee_codes')
    
#     # Fallback to regular hot containers function
#     return get_hot_containers(query)
# ...existing code...


# Make the consignee-specific entry point a plain alias to the unified function
def get_hot_containers_by_consignee(query: str) -> str:
    """
    Alias to get_hot_containers. Consignee scoping comes from _df() via thread-local codes.
    """
    return get_hot_containers(query)

# ...existing code...

def check_transit_status(query: str) -> str:
    """Question 14: Check if cargo/PO is currently in transit"""
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
    import re
    
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

# ...existing code...

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
    

# ...existing code...

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
    

def get_containers_by_supplier(query: str) -> str:
    """Entry point used by router; dispatch to in-transit or last-days."""
    ql = query.lower()
    if 'transit' in ql:
        return get_supplier_in_transit(query)
    # treat "last/past N days" as arrived listing
    if re.search(r'(?:last|past)\s+\d{1,3}\s+days?', ql):
        return get_supplier_last_days(query)
    # default to in-transit if user didn’t specify
    return get_supplier_in_transit(query)




# ...existing code...
# ...existing code...

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
    Robust to tokens like 'PO5302816722' vs '5302816722'.
    """
    if pd.isna(cell) or not po_norm:
        return False
    parts = re.split(r'[,;/\|\s]+', str(cell))
    q_digits = po_norm.isdigit()
    q_strip_po = po_norm[2:] if po_norm.startswith('PO') else po_norm
    for p in parts:
        tk = _normalize_po_token(p)
        if tk == po_norm:
            return True
        # If query is all digits, allow suffix match of dataset token
        if q_digits and tk.endswith(po_norm):
            return True
        # If query has 'PO' prefix, allow match against numeric tail in dataset token
        if po_norm.startswith('PO') and (tk == q_strip_po or tk.endswith(q_strip_po)):
            return True
    return False

# ...existing code...

# ...existing code...
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
# ------------------------------------------------------------------
# 1️⃣ Container Milestones
# ------------------------------------------------------------------
def get_container_milestones(input_str: str) -> str:
    """
    Retrieve all milestone dates for a specific container, sorted chronologically.
    Input: Provide a valid container number (partial or full).
    Output: List of milestone events and their dates for the container.
    If no container is found, prompts for a valid container number.
    """
    container_no = extract_container_number(input_str)
    if not container_no:
        return "Please specify a valid container number."

    df = _df()
    df["container_number"] = df["container_number"].astype(str)

    # Exact match after normalising
    clean = clean_container_number(container_no)
    rows = df[df["container_number"].str.replace(" ", "").str.upper() == clean]

    # Fallback to contains-match
    if rows.empty:
        #rows = df[df["container_number"].str.contains(container_no, case=False, na=False)]
        rows = df[
            df["po_number_multiple"].str.contains(container_no, case=False, na=False) |
            df["ocean_bl_no_multiple"].str.contains(container_no, case=False, na=False)
            ]

    if rows.empty:
        return f"No data found for container {container_no}."

    row = rows.iloc[0]

    # -------------------------------------------------
    # Build the milestone list (only keep non-null dates)
    # -------------------------------------------------
    
    row_milestone_map = [
            ("<strong>Departed From</strong>", row.get("load_port"), safe_date(row.get("atd_lp"))),
            ("<strong>Arrived at Final Load Port</strong>", row.get("final_load_port"), safe_date(row.get("ata_flp"))),
            ("<strong>Departed from Final Load Port</strong>", row.get("final_load_port"), safe_date(row.get("atd_flp"))),
            ("<strong>Expected at Discharge Port</strong>", row.get("discharge_port"), safe_date(row.get("derived_ata_dp"))),
            ("<strong>Reached at Discharge Port</strong>", row.get("discharge_port"), safe_date(row.get("ata_dp"))),
            ("<strong>Reached at Last CY</strong>", row.get("last_cy_location"), safe_date(row.get("equipment_arrived_at_last_cy"))),
            ("<strong>Out Gate at Last CY</strong>", row.get("out_gate_at_last_cy_lcn"), safe_date(row.get("out_gate_at_last_cy"))),
            ("<strong>Delivered at</strong>", row.get("delivery_date_to_consignee_lcn"), safe_date(row.get("delivery_date_to_consignee"))),
            ("<strong>Empty Container Returned to</strong>", row.get("empty_container_return_lcn"), safe_date(row.get("empty_container_return_date"))),
        ]
    

    c_df = pd.DataFrame(row_milestone_map)
    # print(c_df)

    f_df = c_df.dropna(subset=2)
    # print(f_df)

    f_line = f_df.iloc[0]
    l_line = f_df.iloc[-1]
    # print(l_line)

    # print("Bot Answer:_____")
    res = f"The Container <con>{container_no}</con> {l_line.get(0)} {l_line.get(1)} on {l_line.get(2)}\n\n <MILESTONE> {f_df.to_string(index=False, header=False)}."
    # print(res)

    # return "\n".join(status_lines)
    return res

def safe_date(val):
    """Return only YYYY-MM-DD or None if NaT/NaN/None/empty."""
    if pd.isna(val):  # catches NaN and NaT
        return None
    return str(val).split()[0]


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
    from fuzzywuzzy import process
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

# ------------------------------------------------------------------
# 2️⃣ Delayed Containers (X days)
# ------------------------------------------------------------------
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

# ...existing code...
# ...existing code...
def get_delayed_containers(question: str = None, consignee_code: str = None, **kwargs) -> str:
    """
    Find containers delayed by specified number of days - supports exact/range queries, location and hot filters.
    Accepts parameters:
      - question: natural language string (preferred)
      - consignee_code: optional filter to restrict to a specific consignee
    Returns: list of record dicts or textual message if none found.
    """
    import re
    import pandas as pd

    query = (question or "")  # support both 'question' and legacy 'query' param
    df = _df()

    # If consignee_code provided, filter early
    if consignee_code and "consignee_code_multiple" in df.columns:
        # match substring - consignees in dataset often like "NAME(CODE)"
        df = df[df["consignee_code_multiple"].astype(str).str.contains(str(consignee_code))]

    # Ensure datetimes with explicit formats to avoid warnings
    # Try common formats, coerce errors
    for col in ["eta_dp", "ata_dp", "predictive_eta_fd", "revised_eta"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    arrived = df[df["ata_dp"].notna()].copy()
    in_transit = df[df["ata_dp"].isna()].copy()

    if arrived.empty and in_transit.empty:
        return "No container records available for your authorized consignees."

    # compute arrived delay days
    if not arrived.empty:
        arrived["delay_days"] = (arrived["ata_dp"] - arrived["eta_dp"]).dt.days
        arrived["delay_days"] = arrived["delay_days"].fillna(0).astype(int)

    # compute predicted delay for in-transit
    if not in_transit.empty:
        pred_col = None
        for c in ("predictive_eta_fd", "revised_eta"):
            if c in in_transit.columns and in_transit[c].notna().any():
                pred_col = c
                break
        if pred_col:
            in_transit["predicted_delay_days"] = (in_transit[pred_col] - in_transit["eta_dp"]).dt.days
            in_transit["predicted_delay_days"] = in_transit["predicted_delay_days"].fillna(0).astype(int)
        else:
            in_transit["predicted_delay_days"] = pd.NA

    # location columns
    port_cols = [c for c in ["discharge_port", "vehicle_arrival_lcn", "final_destination", "place_of_delivery", "load_port"] if c in df.columns]

    def _extract_loc_code_and_name(q: str, frame):
        q_up = (q or "").upper()
        # code in parentheses
        m = re.search(r"\(([A-Z0-9]{3,6})\)", q_up)
        if m:
            return m.group(1), None
        # detect known codes present in port columns
        cand_codes = set(re.findall(r"\b[A-Z0-9]{3,6}\b", q_up))
        if port_cols and cand_codes:
            known_codes = set()
            for c in port_cols:
                vals = frame[c].dropna().astype(str).str.upper()
                known_codes |= set(re.findall(r"\(([A-Z0-9]{3,6})\)", " ".join(vals.tolist())))
            for code in cand_codes:
                if code in known_codes:
                    return code, None
        m2 = re.search(r"(?:\bat|\bin|\bto|\bfrom)\s+([A-Z][A-Z\s\.,'\-]{2,})$", q_up)
        name = m2.group(1).strip() if m2 else None
        return None, name

    code, name = _extract_loc_code_and_name(query, df)

    def _apply_location_mask(frame, code, name):
        if frame is None or frame.empty:
            return frame
        if not (code or name):
            return frame
        loc_mask = pd.Series(False, index=frame.index)
        if code:
            for c in port_cols:
                loc_mask |= frame[c].astype(str).str.upper().str.contains(rf"\({re.escape(code)}\)", na=False)
        else:
            tokens = [w for w in re.split(r"\W+", name or "") if len(w) >= 3]
            if not tokens:
                return frame
            for c in port_cols:
                col_vals = frame[c].astype(str).str.upper()
                cond = pd.Series(True, index=frame.index)
                for t in tokens:
                    cond &= col_vals.str.contains(re.escape(t), na=False)
                loc_mask |= cond
        return frame[loc_mask].copy()

    arrived = _apply_location_mask(arrived, code, name)
    in_transit = _apply_location_mask(in_transit, code, name)

    if arrived.empty and in_transit.empty:
        where = f"{code or name}"
        return f"No delayed/early containers at {where} for your authorized consignees."

    q = (query or "").lower()

    # weeks -> days conversion
    def _weeks_to_days(m):
        txt = m.group(1)
        if re.search(r'\b(a|one)\b', txt):
            return "7 days"
        mn = re.search(r'(\d+)', txt)
        if mn:
            return f"{int(mn.group(1))*7} days"
        return "7 days"

    q = re.sub(r'\b((?:a|one|\d+)\s+weeks?)\b', lambda m: _weeks_to_days(m), q, flags=re.IGNORECASE)

    significant_threshold = 7
    is_significant = bool(re.search(r"\bsignificant\b|\bmajor\b|\bsevere\b", q))
    hot_only = bool(re.search(r"\bhot\b|\bpriority\b|\bhigh[-\s]?priority\b", q))
    wants_predicted = bool(re.search(r"not going to make|won't make|will not make|unlikely to make", q))

    # detect ranges and numbers
    range_dash = re.search(r"\b(\d+)\s*[-–—]\s*(\d+)\s*days?", q)
    between = re.search(r"(?:between|from)\s*(\d+)\s*(?:and|to|-)\s*(\d+)\s*days?", q)
    strictly_less = re.search(r"(?:less\s+than|under|below|<)\s*(\d+)\s*days?", q)
    up_to = re.search(r"(?:up\s*to|within|no\s*more\s*than|<=)\s*(\d+)\s*days?", q)
    more_than = re.search(r"(?:more\s+than|over|>\s*)(\d+)\s*days?", q)
    at_least = re.search(r"(?:at\s+least|>=|minimum)\s*(\d+)\s*days?", q)
    or_more = re.search(r"\b(\d+)\s*days?\s*(?:or\s+more|or\s+above|and\s+more)\b", q)
    exact_ph = re.search(r"(?:delayed|late|overdue|behind|missed)\s+by\s+(\d+)\s*days?", q)
    plain_days = re.search(r"\b(\d+)\s*days?\b", q)

    low = None
    high = None
    query_type = None

    if range_dash or between:
        m = range_dash or between
        d1, d2 = int(m.group(1)), int(m.group(2))
        low, high = min(d1, d2), max(d1, d2)
        query_type = f"between {low} and {high}"
    elif strictly_less:
        days = int(strictly_less.group(1))
        low, high = 1, days-1 if days>1 else 0
        query_type = f"less than {days}"
    elif up_to:
        days = int(up_to.group(1))
        low, high = 1, days
        query_type = f"up to {days}"
    elif more_than:
        days = int(more_than.group(1))
        low, high = days+1, None
        query_type = f"more than {days}"
    elif at_least or or_more:
        m = at_least or or_more
        days = int(m.group(1))
        low, high = days, None
        query_type = f"at least {days}"
    elif exact_ph or (plain_days and ("late" in q or "delayed" in q or "overdue" in q or "behind" in q or "missed" in q)):
        days = int((exact_ph or plain_days).group(1))
        low, high = days, days
        query_type = f"exactly {days}"
    else:
        if is_significant:
            low, high = significant_threshold, None
            query_type = f"significant (≥{significant_threshold})"
        else:
            query_type = "more than 0"

    results = []
    def _matches(v):
        if v is pd.NA or pd.isna(v):
            return False
        try:
            vi = int(v)
        except Exception:
            return False
        if low is not None and vi < low:
            return False
        if high is not None and vi > high:
            return False
        return True

    # arrived checks
    if not arrived.empty:
        asks_early = bool(re.search(r"\bearly\b|\bahead of schedule\b", q))
        if low is not None or high is not None:
            sel = arrived[arrived["delay_days"].apply(lambda v: _matches(v))]
        else:
            if asks_early:
                sel = arrived[arrived["delay_days"] < 0].copy()
            else:
                sel = arrived[arrived["delay_days"] > 0].copy()
        if not sel.empty:
            sel["status"] = sel["delay_days"].apply(lambda d: "arrived_late" if d>0 else ("arrived_early" if d<0 else "arrived_on_time"))
            sel["effective_delay_days"] = sel["delay_days"]
            results.append(sel)

    # in-transit checks
    if not in_transit.empty:
        if "predicted_delay_days" in in_transit.columns and in_transit["predicted_delay_days"].notna().any():
            if low is not None or high is not None:
                selp = in_transit[in_transit["predicted_delay_days"].apply(lambda v: _matches(v))]
            else:
                asks_early = bool(re.search(r"\bearly\b|\bahead of schedule\b", q))
                if asks_early:
                    selp = in_transit[in_transit["predicted_delay_days"] < 0].copy()
                else:
                    selp = in_transit[in_transit["predicted_delay_days"] > 0].copy()
            if not selp.empty:
                selp["status"] = selp["predicted_delay_days"].apply(lambda d: "predicted_late" if d>0 else ("predicted_early" if d<0 else "predicted_on_time"))
                selp["effective_delay_days"] = selp["predicted_delay_days"]
                results.append(selp)

    if not results:
        where = f" at {code or name}" if (code or name) else ""
        return f"No containers match the delay/early criteria ({query_type}) for your authorized consignees{where}."

    combined = pd.concat(results, axis=0, ignore_index=False)

    # hot filter
    if hot_only and "hot_container_flag" in combined.columns:
        combined = combined[combined["hot_container_flag"].astype(bool)]
        if combined.empty:
            where = f" at {code or name}" if (code or name) else ""
            return f"No hot containers match the delay/early criteria ({query_type}) for your authorized consignees{where}."

    # select columns
    cols = ["container_number", "eta_dp", "ata_dp", "effective_delay_days", "status", "consignee_code_multiple","hot_container_flag"]
    cols += [c for c in ["discharge_port", "vehicle_arrival_lcn", "final_destination", "place_of_delivery", "load_port"] if c in combined.columns]
    if "hot_container_flag" in combined.columns:
        cols.append("hot_container_flag")
    cols = [c for i,c in enumerate(cols) if c in combined.columns and c not in cols[:i]]

    out = combined[cols].copy()
    for d in ["eta_dp","ata_dp"]:
        if d in out.columns and pd.api.types.is_datetime64_any_dtype(out[d]):
            out[d] = out[d].dt.strftime("%Y-%m-%d")

    out = out.sort_values(["effective_delay_days"], ascending=False)
    out = out.rename(columns={"effective_delay_days":"delay_days"})
    return out.reset_index(drop=True).to_dict(orient="records")


def get_delayed_containers(query: str) -> str:
    """Find containers delayed by specified number of days - supports exact or range queries.
       Note: less than/under/below/< N → 0 < delay_days < N; up to/within/no more than/<= N → 0 < delay_days ≤ N.
    """
    import re
    df = _df()
    df = ensure_datetime(df, ["eta_dp", "ata_dp"])
 
    arrived = df[df["ata_dp"].notna()].copy()
    if arrived.empty:
        return "No containers have arrived for your authorized consignees."
 
    arrived["delay_days"] = (arrived["ata_dp"] - arrived["eta_dp"]).dt.days
    arrived["delay_days"] = arrived["delay_days"].fillna(0).astype(int)
 
    port_cols = [c for c in ["discharge_port", "vehicle_arrival_lcn", "final_destination", "place_of_delivery"] if c in arrived.columns]
    def _extract_loc_code_and_name(q: str):
        q_up = (q or "").upper()
        m = re.search(r"\(([A-Z0-9]{3,6})\)", q_up)
        if m: return m.group(1), None
        cand_codes = set(re.findall(r"\b[A-Z0-9]{3,6}\b", q_up))
        if port_cols and cand_codes:
            known_codes = set()
            for c in port_cols:
                vals = arrived[c].dropna().astype(str).str.upper()
                known_codes |= set(re.findall(r"\(([A-Z0-9]{3,6})\)", " ".join(vals.tolist())))
            for code in cand_codes:
                if code in known_codes:
                    return code, None
        m2 = re.search(r"(?:\bat|\bin|\bto)\s+([A-Z][A-Z\s\.,'-]{3,})$", q_up)
        name = m2.group(1).strip() if m2 else None
        return None, name
 
    code, name = _extract_loc_code_and_name(query)
    if code or name:
        loc_mask = pd.Series(False, index=arrived.index)
        if code:
            for c in port_cols:
                loc_mask |= arrived[c].astype(str).str.upper().str.contains(rf"\({re.escape(code)}\)", na=False)
        else:
            tokens = [w for w in re.split(r"\W+", name or "") if len(w) >= 3]
            for c in port_cols:
                col_vals = arrived[c].astype(str).str.upper()
                cond = pd.Series(True, index=arrived.index)
                for t in tokens:
                    cond &= col_vals.str.contains(re.escape(t), na=False)
                loc_mask |= cond
        arrived = arrived[loc_mask].copy()
        if arrived.empty:
            where = f"{code or name}"
            return f"No delayed containers at {where} for your authorized consignees."
 
    q = query.lower()
 
    # Strict buckets
    strictly_less_match = re.search(r"(?:less\s+than|under|below|<)\s*(\d+)\s*days?", q, re.IGNORECASE)
    up_to_match         = re.search(r"(?:up\s*to|within|no\s*more\s*than|<=\s*)(\d+)\s*days?", q, re.IGNORECASE)
    more_than_match     = re.search(r"(?:(?:delayed|late|overdue|behind)\s+by\s+)?(?:more than|over|>\s*)\s*(\d+)\s*days?", q, re.IGNORECASE)
    at_least_explicit   = re.search(r"(?:(?:delayed|late|overdue|behind)\s+by\s+)?(?:at\s+least|>=\s*|greater\s+than\s+or\s+equal\s+to|minimum)\s*(\d+)\s*days?", q, re.IGNORECASE)
    or_more_match       = re.search(r"\b(\d+)\s+days?\s*(?:or\s+more|and\s+more|or\s+above)\b", q, re.IGNORECASE)
    exact_phrase_match  = re.search(r"(?:delayed|late|overdue|behind)\s+by\s+(\d+)\s+days?", q, re.IGNORECASE)
    plain_days_match    = re.search(r"\b(\d+)\s+days?\b(?:\s*(?:late|delayed|overdue|behind))?", q, re.IGNORECASE)
 
    if strictly_less_match:
        days = int(strictly_less_match.group(1))
        delayed_df = arrived[(arrived["delay_days"] > 0) & (arrived["delay_days"] < days)]
        query_type = f"less than {days}"
    elif up_to_match:
        days = int(up_to_match.group(1))
        delayed_df = arrived[(arrived["delay_days"] > 0) & (arrived["delay_days"] <= days)]
        query_type = f"up to {days}"
    elif more_than_match:
        days = int(more_than_match.group(1))
        delayed_df = arrived[arrived["delay_days"] > days]
        query_type = f"more than {days}"
    elif at_least_explicit or or_more_match:
        days = int((at_least_explicit or or_more_match).group(1))
        delayed_df = arrived[arrived["delay_days"] >= days]
        query_type = f"at least {days}"
    elif exact_phrase_match or plain_days_match:
        days = int((exact_phrase_match or plain_days_match).group(1))
        delayed_df = arrived[arrived["delay_days"] == days]
        query_type = f"exactly {days}"
    else:
        delayed_df = arrived[arrived["delay_days"] > 0]
        query_type = "more than 0"
 
    delayed_df = delayed_df[delayed_df["delay_days"] > 0]
    if delayed_df.empty:
        where = f" at {code or name}" if (code or name) else ""
        return f"No containers are delayed by {query_type} days for your authorized consignees{where}."
 
    cols = ["container_number", "eta_dp", "ata_dp", "delay_days", "consignee_code_multiple", "discharge_port","hot_container_flag"]
    if "vehicle_arrival_lcn" in delayed_df.columns:
        cols.append("vehicle_arrival_lcn")
    cols = [c for c in cols if c in delayed_df.columns]
    out = delayed_df[cols].sort_values("delay_days", ascending=False).copy()
 
    if "eta_dp" in out.columns and pd.api.types.is_datetime64_any_dtype(out["eta_dp"]):
        out["eta_dp"] = out["eta_dp"].dt.strftime("%Y-%m-%d")
    if "ata_dp" in out.columns and pd.api.types.is_datetime64_any_dtype(out["ata_dp"]):
        out["ata_dp"] = out["ata_dp"].dt.strftime("%Y-%m-%d")
 
    return out.to_dict(orient="records")

# ...existing code...


def get_hot_containers(query: str) -> str:
    """
    Unified hot-container handler.

    Supports:
      - which hot containers are delayed/late at <PORT>
      - which hot containers are delayed/late by X days at <PORT>
      - which hot containers are delayed less than / under / below X days
      - which hot containers are delayed between X–Y days
      - which hot containers missed ETA or are behind schedule
      - which hot containers will arrive at <PORT>
      - which hot containers will arrive at <PORT> in next X days
      - "List hot shipments from <location> that are late by 5+ days"
      - "Highlight hot shipments that missed ETA by 1–3 days"
      - "Show me hot cargoes that missed ETA by more than [number] days"

    Notes:
    - Consignee scoping comes from _df() (thread-local).
    - Strict port filter: rows whose discharge_port OR vehicle_arrival_lcn contains '(CODE)'.
    """
    import re
    import pandas as pd

    df = _df()  # already consignee-filtered if set

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

    # Location filters
    port_cols = [c for c in ["discharge_port", "vehicle_arrival_lcn", "final_destination", "place_of_delivery", "load_port"]
                 if c in hot_df.columns]

    def _extract_loc_code_and_name(q: str):
        q_up = (q or "").upper()
        m = re.search(r"\(([A-Z0-9]{3,6})\)", q_up)
        if m:
            return m.group(1), None
        # known code in dataset
        cand_codes = set(re.findall(r"\b[A-Z0-9]{3,6}\b", q_up))
        if port_cols and cand_codes:
            known_codes = set()
            for c in port_cols:
                vals = hot_df[c].dropna().astype(str).str.upper()
                known_codes |= set(re.findall(r"\(([A-Z0-9]{3,6})\)", " ".join(vals.tolist())))
            for code in cand_codes:
                if code in known_codes:
                    return code, None
        m2 = re.search(r"(?:\bat|\bin|to|from)\s+([A-Z][A-Z\s\.,'-]{2,})$", q_up)
        name = m2.group(1).strip() if m2 else None
        return None, name

    code, name = _extract_loc_code_and_name(query)
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

    # A) Delayed/late/missed ETA hot containers (arrived only)
    if any(w in ql for w in ("delay", "late", "overdue", "behind", "missed", "eta", "deadline")):
        hot_df = ensure_datetime(hot_df, ["eta_dp", "ata_dp"])
        arrived = hot_df[hot_df["ata_dp"].notna()].copy()
        if arrived.empty:
            where = f" at {code or name}" if (code or name) else ""
            return f"No hot containers have arrived{where} for your authorized consignees."

        arrived["delay_days"] = (arrived["ata_dp"] - arrived["eta_dp"]).dt.days.fillna(0).astype(int)

        # ------------------------------
        # Numeric pattern parsing
        # ------------------------------
        range_match = re.search(r"(\d+)\s*[-–—]\s*(\d+)\s*days?", ql)
        less_than = re.search(r"(?:less\s+than|under|below|<)\s*(\d+)\s*days?", ql)
        more_than = re.search(r"(?:more\s+than|over|>\s*)(\d+)\s*days?", ql)
        plus_sign = re.search(r"\b(\d+)\s*\+\s*days?\b", ql)  # ✅ detect 8+ days
        exact     = re.search(r"(?:by|of|in)\s+(\d+)\s+days?", ql)

        # apply range logic
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
        # ------------------------------

        delayed = delayed[delayed[hot_flag_col].apply(_is_hot)]
        delayed = delayed[delayed["delay_days"] > 0]
        if delayed.empty:
            where = f" at {code or name}" if (code or name) else ""
            return f"No hot containers are delayed for your authorized consignees{where}."

        cols = ["container_number", "eta_dp", "ata_dp", "delay_days", "discharge_port","hot_container_flag",'consignee_code_multiple']
        if "vehicle_arrival_lcn" in delayed.columns:
            cols.append("vehicle_arrival_lcn")
        cols = [c for c in cols if c in delayed.columns]

        out = delayed[cols].sort_values("delay_days", ascending=False).head(100).copy()

        for dcol in ["eta_dp", "ata_dp"]:
            if dcol in out.columns and pd.api.types.is_datetime64_any_dtype(out[dcol]):
                out[dcol] = out[dcol].dt.strftime("%Y-%m-%d")

        return out.where(pd.notnull(out), None).to_dict(orient="records")


    # C) Fallback: simple hot listing
    display_cols = ['container_number', 'consignee_code_multiple']
    display_cols += [c for c in ['discharge_port', 'eta_dp', 'revised_eta','hot_container_flag','consignee_code_multiple'] if c in hot_df.columns]
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





# ...existing code...
def get_upcoming_arrivals(query: str) -> str:
    """
    List containers scheduled to arrive within the next X days.
    - Parses several natural forms for "next N days".
    - Uses eta column (eta_dp preferred) and excludes rows with ata_dp (already arrived).
    - Respects consignee filtering via _df().
    Returns list of dict records (up to 50 rows) or a short message if none found.
    """
    # parse days
    patterns = [
        r"(?:next|upcoming|in)\s+(\d+)\s+days?",  # "next 2 days", "in 2 days"
        r"(\d+)\s+days?",                         # "2 days"
        r"arriving.*?(\d+)\s+days?",              # "arriving in 2 days"
        r"will.*?arrive.*?(\d+)\s+days?",         # "will arrive in 2 days"
    ]
    days = None
    for p in patterns:
        m = re.search(p, query, re.IGNORECASE)
        if m:
            days = int(m.group(1))
            break
    if days is None:
        days = 7
# ...existing code...
    df = _df()  # respects consignee filtering

    # apply transport mode filter if present (normalize column and match)
    modes = extract_transport_modes(query)
    if modes and 'transport_mode' in df.columns:
        df = df[df['transport_mode'].astype(str).str.lower().apply(lambda s: any(m in s for m in modes))]

    # Choose ETA-like column
    eta_col = next((c for c in ["revised_eta", "eta_dp", "eta", "eta_fd", "estimated_time_arrival"] if c in df.columns), None)
    if not eta_col:
        return "ETA column not found in the data."

    # Ensure relevant date columns are datetime on the filtered df
    parse_cols = [eta_col]
    if "ata_dp" in df.columns:
        parse_cols.append("ata_dp")
    df = ensure_datetime(df, parse_cols)

    today = pd.Timestamp.today().normalize()
    future = today + pd.Timedelta(days=days)

    # Build mask safely on the already-filtered df
    mask = (df[eta_col].notna()) & (df[eta_col] >= today) & (df[eta_col] <= future)
    if "ata_dp" in df.columns:
        mask &= df["ata_dp"].isna()  # exclude already-arrived

    upcoming = df[mask].copy()
# ...existing code...

    upcoming = df[mask].copy()
    if upcoming.empty:
        return f"No containers scheduled to arrive in the next {days} days for your authorized consignees."

    # Prepare output
    cols = ["container_number", "discharge_port", eta_col]
    if "consignee_code_multiple" in upcoming.columns:
        cols.append("consignee_code_multiple")
    cols = [c for c in cols if c in upcoming.columns]

    upcoming = upcoming[cols].head(50).copy()
    # format dates
    if eta_col in upcoming.columns and pd.api.types.is_datetime64_any_dtype(upcoming[eta_col]):
        upcoming[eta_col] = upcoming[eta_col].dt.strftime("%Y-%m-%d")

    return upcoming.to_dict(orient="records")
# ...existing code...

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


import re
from difflib import get_close_matches
import pandas as pd

# ...existing code...
def get_arrivals_by_port(query: str) -> str:
    """
    Find containers arriving at a specific port or country within the next N days.
    Behaviour:
    - Parse 'next X days' robustly.
    - For each row, prefer revised_eta; if null use eta_dp (eta_for_filter).
    - Exclude rows that already have ata_dp (already arrived).
    - Return up to 50 matching rows (dict records) with formatted dates.
    """
    
    df = _df()

    filtered = df[mask].copy()

    # apply transport mode filter if present
    modes = extract_transport_modes(query)
    if modes and 'transport_mode' in filtered.columns:
        filtered = filtered[filtered['transport_mode'].astype(str).str.lower().apply(lambda s: any(m in s for m in modes))]

    # ---------- 1) Parse timeframe ----------
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

    today = pd.Timestamp.today().normalize()
    end_date = today + pd.Timedelta(days=n_days)

    # <-- ADDED: log parsed timeframe for debugging
    try:
        logger.info(f"[get_arrivals_by_port] query={query!r} parsed_n_days={n_days} today={today.strftime('%Y-%m-%d')} end_date={end_date.strftime('%Y-%m-%d')}")
    except Exception:
        print(f"[get_arrivals_by_port] parsed_n_days={n_days} today={today} end_date={end_date}")

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

    # Inclusive window and exclude already-arrived
    date_mask = (filtered['eta_for_filter'] >= today) & (filtered['eta_for_filter'] <= end_date)
    if 'ata_dp' in filtered.columns:
        date_mask &= filtered['ata_dp'].isna()

    arrivals = filtered[date_mask].copy()
    if arrivals.empty:
        return (
            f"No containers with ETA between {today.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')} "
            f"for the requested port ('{port_code_query or port_name_query}')."
        )

    # ---------- 6) Build display ----------
    display_cols = ['container_number', 'discharge_port', 'revised_eta', 'eta_dp']
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

    result_df = arrivals[display_cols].sort_values('eta_for_filter').head(50).copy()

    # Format date columns
    for dcol in ['revised_eta', 'eta_dp', 'eta_for_filter']:
        if dcol in result_df.columns and pd.api.types.is_datetime64_any_dtype(result_df[dcol]):
            result_df[dcol] = result_df[dcol].dt.strftime('%Y-%m-%d')

    # Drop internal helper column from final output
    if 'eta_for_filter' in result_df.columns:
        result_df = result_df.drop(columns=['eta_for_filter'])

    result_data = result_df.where(pd.notnull(result_df), None)
    
    return result_data.to_dict(orient="records")
# ...existing code...



# ------------------------------------------------------------------
# 6️⃣ Keyword / fuzzy search
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# 7️⃣ PandasAI data analysis
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# 8️⃣ Field information (generic) – the biggest function in the original script
# ------------------------------------------------------------------
def get_field_info(query: str) -> str:
    """
    Retrieve detailed information for a container or summarize a specific field.
    Input: Query can be a container number or mention a field (port, vessel, carrier, date, location).
    Output: For containers, shows all non-null fields. For fields, summarizes top values or date range.
    Uses fuzzy matching for ambiguous queries.
    """
    from fuzzywuzzy import process

    df = _df()
    container_no = extract_container_number(query)

    # ------------------------------------------------------------------
    # a) If a container number is present → show its fields
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------

    # b) No container – try to infer which *type* of field the user wants
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # c) Determine column list based on detected field type or fuzzy match
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # d) Show a concise summary (top 5 values / date range)
    # ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# 9️⃣ Vessel Info (tiny helper that was separate in the original script)
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# 10️⃣ Upcoming PO's (by ETD window, ATA null)
# ------------------------------------------------------------------
def get_upcoming_pos(query: str) -> str:
    """
    List PO's scheduled to ship in the next X days (ETD window, not yet shipped).
    Input: Query should specify 'next <number> days' or 'coming <number> days'.
    Output: Table of upcoming PO's with ETD and destination details.
    Defaults to 7 days if not specified.
    """
    m = re.search(r"(?:next|coming)\s+(\d+)\s+days", query, re.IGNORECASE)
    days = int(m.group(1)) if m else 10

    df = _df()
    etd_cols = [c for c in ["etd_lp", "etd_flp"] if c in df.columns]
    if not etd_cols:
        return "ETD columns not found – cannot compute upcoming POs."

    # ensure dates are datetime
    for c in etd_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    today = pd.Timestamp.today().normalize()
    future = today + pd.Timedelta(days=days)

    mask = pd.Series(False, index=df.index)
    for c in etd_cols:
        mask |= (df[c] >= today) & (df[c] <= future)

    # ATA must be null (meaning the PO hasn't been shipped yet)
    ata_cols = [c for c in ["ata_lp", "ata_flp"] if c in df.columns]
    if ata_cols:
        for a in ata_cols:
            mask &= df[a].isna()

    filtered = df[mask]
    if filtered.empty:
        return f"No upcoming POs in the next {days} days."

    po_col = "po_number_multiple" if "po_number_multiple" in df.columns else "po_number"
    cols = [po_col] + etd_cols + ["discharge_port", "final_destination"]
    cols = [c for c in cols if c in filtered.columns]

    #result = filtered[cols].drop_duplicates().to_string(index=False)
    #return f"Upcoming POs (next {days} days):\n\n{result}"
    return filtered[cols].drop_duplicates().head(15).to_dict(orient="records")


# ------------------------------------------------------------------
# 11️⃣ Delayed PO's (complex ETA / milestone logic)
# ------------------------------------------------------------------
def get_delayed_pos(_: str = "") -> str:
    """
    Find PO's that are delayed based on delivery/empty-container status and ETA logic.
    Input: No specific input required.
    Output: Table of delayed PO's with ETA, delivery, and return details.
    Uses predictive/revised ETA and milestone logic.
    """
    df = _df()
    po_col = "po_number_multiple" if "po_number_multiple" in df.columns else "po_number"
    if po_col not in df.columns:
        return "PO column missing from data."

    # ------------------------------------------------------------------
    # Ensure all date columns are datetime
    # ------------------------------------------------------------------
    date_cols = ["delivery_date_to_consignee", "empty_container_return_date",
                 "predictive_eta_fd", "revised_eta_fd", "eta_fd"]
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    not_delivered = df["delivery_date_to_consignee"].isna() if "delivery_date_to_consignee" in df.columns else pd.Series(True, index=df.index)
    not_returned = df["empty_container_return_date"].isna() if "empty_container_return_date" in df.columns else pd.Series(True, index=df.index)

    # NVL logic – use predictive if present, else revised
    if "predictive_eta_fd" in df.columns and "revised_eta_fd" in df.columns:
        df["next_eta"] = df["predictive_eta_fd"].fillna(df["revised_eta_fd"])
    elif "predictive_eta_fd" in df.columns:
        df["next_eta"] = df["predictive_eta_fd"]
    elif "revised_eta_fd" in df.columns:
        df["next_eta"] = df["revised_eta_fd"]
    else:
        df["next_eta"] = pd.NaT

    today = pd.Timestamp.today().normalize()
    future_eta = df["next_eta"] > today
    past_eta = (df["eta_fd"] < today) & df["eta_fd"].notna()

    mask = (not_delivered & not_returned) & (future_eta | past_eta)
    delayed = df[mask]

    if delayed.empty:
        return "No delayed PO's found based on the current criteria."

    show = [po_col, "eta_fd", "predictive_eta_fd", "revised_eta_fd",
            "delivery_date_to_consignee", "empty_container_return_date"]
    show = [c for c in show if c in delayed.columns]

    result = delayed[show].drop_duplicates().head(15).to_string(index=False)
    return f"Delayed PO's:\n\n{result}"


# ------------------------------------------------------------------
# 12️⃣ Containers arriving soon (ETA window & ATA null)
# ------------------------------------------------------------------
# ...existing code...
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
# ...existing code...


def check_arrival_status(input_str: str) -> str:
    """
    Check if a container or PO has arrived - now with consignee filtering.
    """
    import re
    from datetime import datetime
    import pandas as pd

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


# def get_container_carrier(input_str: str) -> str:
#     """
#     Get the carrier information for a specific container or PO.
#     Input: Query should mention a container number or PO number (partial or full).
#     Output: Carrier details including carrier name, code, and SCAC code if available.
#     For PO queries: Shows final carrier. If multiple Show latest w.r.t. ETD/ETA.
#     If no container/PO is found, prompts for a valid identifier.
#     """
#     # Try to extract container number first
#     container_no = extract_container_number(input_str)

#     # Try to extract PO number if no container found
#     po_no = None
#     if not container_no:
#         po_no = extract_po_number(input_str)

#     if not container_no and not po_no:
#         return "Please specify a valid container number or PO number to get carrier information."
# ...existing code...
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

    # ...existing code...
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
    from services.azure_blob import get_shipment_df
    from sqlalchemy import create_engine

    df = get_shipment_df().copy()
    engine = create_engine("sqlite:///shipment_blob.db", echo=False, future=True)
    with engine.begin() as conn:
        df.to_sql("shipment", conn, if_exists="replace", index=False)
    return engine

# def get_blob_sql_engine():
#     """
#     Loads the shipment CSV from Azure Blob and creates a persistent SQLite engine for SQL queries.
#     """
#     from agents.azure_agent import get_persistent_sql_engine
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
        from agents.azure_agent import initialize_sql_agent
        
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
# ...existing code...

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

# ...existing code...
# def get_containers_by_transport_mode(query: str) -> str:
#     """
#     Handle queries about transport_mode (e.g. "containers arrived by sea",
#     "which container will arrive by air in next 3 days").
#     Behaviour:
#     - Detect transport mode(s) from query using extract_transport_modes().
#     - If query contains a next/N-days window -> return upcoming arrivals filtered by transport_mode.
#     - Otherwise treat as 'arrived' request and return rows with ata_dp not null filtered by transport_mode.
#     - Uses _df() (consignee filtering), ensure_datetime, and per-row ETA logic where needed.
#     """
#     modes = extract_transport_modes(query)
#     if not modes:
#         return "No transport mode detected in the query."

#     df = _df()
#     if 'transport_mode' not in df.columns:
#         return "No 'transport_mode' column in data."

#     # filter by transport mode (case-insensitive substring match)
#     df_mode = df[df['transport_mode'].astype(str).str.lower().apply(lambda s: any(m in s for m in modes))].copy()
#     if df_mode.empty:
#         return f"No containers found for transport mode(s): {', '.join(sorted(modes))} for your authorized consignees."

#     # Check if query is about future arrivals or past arrivals
#     is_future_query = any(word in query.lower() for word in ["will arrive", "arriving", "upcoming", "next", "future"])
#     m_days = re.search(r'(?:next|in|upcoming|within)\s+(\d{1,3})\s+days?', query, re.IGNORECASE)
    
#     # If user asked for upcoming window -> delegate to arriving-soon logic here (but operate on df_mode)
#     if m_days or is_future_query:
#         days = int(m_days.group(1)) if m_days else 7
#         # use per-row ETA preference
#         # ...rest of the existing code for upcoming arrivals...
    
#     # Otherwise treat as "arrived by <mode>" -> return rows with ata_dp not null
#     if 'ata_dp' not in df_mode.columns:
#         return "No ATA column (ata_dp) present to determine arrived containers."
        
#     df_mode = ensure_datetime(df_mode, ['ata_dp', 'revised_eta', 'eta_dp'])
#     # Explicitly filter for rows where ata_dp is not null (container has arrived)
#     arrived = df_mode[df_mode['ata_dp'].notna()].copy()
    
#     if arrived.empty:
#         return f"No containers have arrived by {', '.join(sorted(modes))} for your authorized consignees."
        
#     # Add po_number_multiple to display columns if available
#     cols = [c for c in ['container_number', 'po_number_multiple', 'discharge_port', 'ata_dp', 'final_carrier_name'] if c in arrived.columns]
#     arrived = arrived[cols].sort_values('ata_dp', ascending=False).head(100).copy()
    
#     if 'ata_dp' in arrived.columns and pd.api.types.is_datetime64_any_dtype(arrived['ata_dp']):
#         arrived['ata_dp'] = arrived['ata_dp'].dt.strftime('%Y-%m-%d')
        
#     return arrived.where(pd.notnull(arrived), None).to_dict(orient='records')




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
    

# Helper: resolve PO from current query or memory; validate against data



# ...existing code...   


# def is_po_hot(query: str) -> str:
#     """
#     Check whether a PO is marked hot via the container's hot flag.
#     Returns a short sentence listing related containers and which are hot.
#     """
    
#     po = extract_po_number(query)
#     if not po:
#         m = re.search(r'\b([A-Z0-9]{6,12})\b', query.upper())
#         po = m.group(1) if m else None
#     if not po:
#         return "Please specify a PO number"

#     po_norm = _normalize_po_token(po)
#     df = _df()
#     po_col = "po_number_multiple" if "po_number_multiple" in df.columns else ("po_number" if "po_number" in df.columns else None)
#     if not po_col:
#         return "PO column not found in the dataset."

#     mask = df[po_col].apply(lambda cell: _po_in_cell(cell, po_norm))
#     matches = df[mask].copy()
#     if matches.empty:
#         return f"No data found for PO {po}."

#     # Identify hot-flag column
#     hot_flag_cols = [c for c in df.columns if 'hot_container_flag' in c.lower()]
#     if not hot_flag_cols:
#         hot_flag_cols = [c for c in df.columns if 'hot_container' in c.lower()]

#     if not hot_flag_cols:
#         return "No hot-container flag column found in the dataset."

#     hot_col = hot_flag_cols[0]

#     def _is_hot(v):
#         if pd.isna(v):
#             return False
#         return str(v).strip().upper() in {"Y", "YES", "TRUE", "1", "HOT"}

#     matches = matches.assign(_is_hot = matches[hot_col].apply(_is_hot))
#     all_containers = sorted(matches["container_number"].dropna().astype(str).unique().tolist())
#     hot_containers = sorted(matches.loc[matches["_is_hot"], "container_number"].dropna().astype(str).unique().tolist())

#     if hot_containers:
#         return f"PO {po} is HOT on container(s): {', '.join(hot_containers)}. Related containers: {', '.join(all_containers)}."
#     else:
#         return f"PO {po} is not marked hot. Related containers: {', '.join(all_containers)}."


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


def is_po_hot(query: str) -> str:
    """
    Check whether a PO is marked hot via the container's hot flag.
    Returns a short sentence listing related containers and which are hot.
    """
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

    matches = matches.assign(_is_hot = matches[hot_col].apply(_is_hot))
    all_containers = sorted(matches["container_number"].dropna().astype(str).unique().tolist())
    hot_containers = sorted(matches.loc[matches["_is_hot"], "container_number"].dropna().astype(str).unique().tolist())

    if hot_containers:
        return f"PO {po} is HOT on container(s): {', '.join(hot_containers)}. Related containers: {', '.join(all_containers)}."
    else:
        return f"PO {po} is not marked hot. Related containers: {', '.join(all_containers)}."

# ...existing code...

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
# ...existing code...

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
# ...existing code...

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
# ...existing code...

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






# ------------------------------------------------------------------
# TOOLS list – must be at module level, not inside any function!
# ------------------------------------------------------------------
TOOLS = [
    Tool(
        name="Get Container Milestones",
        func=get_container_milestones,
        description="Retrieve all milestone dates for a specific container."
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
        description="List containers scheduled to arrive within the next X days."
    ),
    Tool(
        name="Get Arrivals By Port",
        func=get_arrivals_by_port,
        description="Find containers arriving at a specific port or country."
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
        description="List PO's scheduled to ship in the next X days."
    ),
    Tool(
        name="Get Delayed POs",
        func=get_delayed_pos,
        description="Find PO's that are delayed based on delivery/empty-container status and ETA logic."
    ),
    Tool(
        name="Get Containers Arriving Soon",
        func=get_containers_arriving_soon,
        description="List containers arriving soon (ETA window, ATA is null)."
    ),
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
        description="Check if a PO/cargo is currently in transit"
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
        name="Get Containers By Supplier", 
        func=get_containers_by_supplier,
        description="Get containers from a specific supplier, either in transit or from recent days"
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
           "Use this tool ONLY if the user asks directly about hot containers "
           "without mentioning any delay, lateness, ETA, or days. "
           "For example: 'Show my hot containers' or 'List all priority shipments'."),
    ),
    Tool(
        name="Get Hot Containers By Consignee",
        func=get_hot_containers_by_consignee,
        description="Get hot containers for specific consignee codes mentioned in the query"
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
]































































