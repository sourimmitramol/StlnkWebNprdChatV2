# agents/tools.py
import logging
import re
from datetime import datetime, timedelta
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.chat_models import AzureChatOpenAI
from config import settings
import pandas as pd
from fuzzywuzzy import process
from langchain.agents import Tool
from services.vectorstore import get_vectorstore
from utils.container import extract_container_number,extract_po_number
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

# In agents/tools.py - Add this helper function at the top:

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




def get_hot_containers(query: str) -> str:
    """
    Get list of hot containers for specified consignee codes.
    Input: Query mentioning hot containers and optionally consignee codes.
    Output: List of hot containers with relevant details.
    """
    df = _df()  # This automatically filters by consignee if in context
   
    # Check if hot container flag column exists
    hot_flag_cols = [col for col in df.columns if 'hot_container_flag' in col.lower()]
    if not hot_flag_cols:
        # Try alternative column names
        hot_flag_cols = [col for col in df.columns if 'hot_container' in col.lower()]
   
    if not hot_flag_cols:
        return "Hot container flag column not found in the data."
   
    hot_flag_col = hot_flag_cols[0]  # Use the first matching column
   
    # Filter for hot containers (assuming flag is 'Y', 'Yes', True, or 1)
    hot_mask = df[hot_flag_col].astype(str).str.upper().isin(['Y', 'YES', 'TRUE', '1'])
    hot_containers = df[hot_mask]
   
    if hot_containers.empty:
        return "No hot containers found for your authorized consignees."
   
    # Select relevant columns for display
    display_cols = ['container_number','po_number_multiple','consignee_code_multiple','supplier_vendor_name']
 
    # Add additional useful columns if they exist
    #optional_cols = ['po_number_multiple', 'discharge_port', 'eta_dp', 'ata_dp', 'load_port', 'final_vessel_name']
    #for col in optional_cols:
        #if col in hot_containers.columns:
            #display_cols.append(col)
   
    # Ensure we only use columns that exist
    display_cols = [col for col in display_cols if col in hot_containers.columns]
   
    # Sort by ETA if available, otherwise by container number
    if 'eta_dp' in hot_containers.columns:
        hot_containers = hot_containers.sort_values('eta_dp')
    else:
        hot_containers = hot_containers.sort_values('container_number')
   
    # Format the response
    result_data = hot_containers[display_cols]  # Limit to 20 results
 
    # Clean datetime columns for display
    for col in result_data.columns:
        if pd.api.types.is_datetime64_dtype(result_data[col]):
            result_data = result_data.copy()
            result_data[col] = result_data[col].dt.strftime('%Y-%m-%d')
   
    if len(result_data) == 0:
        return "No hot containers found for your authorized consignees."
   
    # Create formatted response
    response_lines = [f"Hot containers found ({len(result_data)} total):"]
   
    for _, row in result_data.iterrows():
        container_info = f"- Container : <con>{row['container_number']}</con>"
       
        if 'consignee_code_multiple' in row:
            container_info += f" | Consignee: {row['consignee_code_multiple']}"
       
        if 'po_number_multiple' in row and pd.notnull(row['po_number_multiple']):
            container_info += f" | PO:<po> {row['po_number_multiple']}</po>"
        if 'ocean_bl_no_multiple' in row and pd.notnull(row['ocean_bl_no_multiple']):
            container_info += f" | PO:<obl> {row['ocean_bl_no_multiple']}</obl>"
       
        if 'discharge_port' in row and pd.notnull(row['discharge_port']):
            container_info += f" | Port: {row['discharge_port']}"
       
        if 'eta_dp' in row and pd.notnull(row['eta_dp']):
            container_info += f" | ETA: {row['eta_dp']}"
        elif 'ata_dp' in row and pd.notnull(row['ata_dp']):
            container_info += f" | Arrived: {row['ata_dp']}"
       
        response_lines.append(container_info)
 
    #return f'The following number of hot containers found: {len(response_lines)}\n' + "\n".join(response_lines)
    return result_data.to_dict(orient="records")

def get_hot_containers_by_consignee(query: str) -> str:
    """
    Get hot containers filtered by specific consignee codes mentioned in the query.
    Input: Query mentioning hot containers and consignee codes.
    Output: Hot containers for specified consignees.
    """
    # Extract consignee codes from query if mentioned
    consignee_pattern = r'consignee[s]?\s+(?:code[s]?\s+)?([0-9,\s]+)'
    consignee_match = re.search(consignee_pattern, query, re.IGNORECASE)
    
    if consignee_match:
        # Parse consignee codes from the query
        codes_str = consignee_match.group(1)
        extracted_codes = [code.strip() for code in re.split(r'[,\s]+', codes_str) if code.strip().isdigit()]
        
        if extracted_codes:
            # Temporarily set these codes in thread context
            import threading
            original_codes = getattr(threading.current_thread(), 'consignee_codes', None)
            threading.current_thread().consignee_codes = extracted_codes
            
            try:
                result = get_hot_containers(query)
                return result
            finally:
                # Restore original codes
                if original_codes:
                    threading.current_thread().consignee_codes = original_codes
                else:
                    if hasattr(threading.current_thread(), 'consignee_codes'):
                        delattr(threading.current_thread(), 'consignee_codes')
    
    # Fallback to regular hot containers function
    return get_hot_containers(query)


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

def get_containers_by_supplier(query: str) -> str:
    """Questions 21-22: Containers from supplier"""
    import re
    
    supplier_match = re.search(r'supplier\s+([A-Z0-9\s]+)', query, re.IGNORECASE)
    if not supplier_match:
        return "Please specify a supplier name."
    
    supplier = supplier_match.group(1).strip()
    df = _df()  # Automatically filters by consignee
    
    supplier_mask = df['supplier_vendor_name'].astype(str).str.contains(supplier, case=False, na=False)
    
    if "transit" in query.lower():
        # Still in transit
        not_delivered = df['delivery_date_to_consignee'].isna()
        not_returned = df['empty_container_return_date'].isna()
        departed = df['atd_lp'].notna()
        
        result = df[supplier_mask & not_delivered & not_returned & departed]
        
        if result.empty:
            return f"No POs from {supplier} currently in transit for your authorized consignees."
        
        cols = ['po_number_multiple', 'supplier_vendor_name', 'eta_fd', 'last_cy_location']
        return f"POs from {supplier} in transit:\n{result[cols].head(10).to_string(index=False)}"
    else:
        # Last X days
        days_match = re.search(r'(\d+)\s+days', query, re.IGNORECASE)
        days = int(days_match.group(1)) if days_match else 30
        
        today = pd.Timestamp.today().normalize()
        start_date = today - pd.Timedelta(days=days)
        
        df = ensure_datetime(df, ['ata_dp'])
        date_mask = (df['ata_dp'] >= start_date) & (df['ata_dp'] <= today)
        
        result = df[supplier_mask & date_mask]
        
        if result.empty:
            return f"No containers from {supplier} in the last {days} days for your authorized consignees."
        
        cols = ['container_number', 'supplier_vendor_name', 'ata_dp', 'discharge_port']
        result = result[cols].head(15)
        result['ata_dp'] = result['ata_dp'].dt.strftime("%Y-%m-%d")
        
        return f"Containers from {supplier} in last {days} days:\n{result.to_string(index=False)}"

def check_po_month_arrival(query: str) -> str:
    """Question 24: Can PO arrive by month end"""
    po_no = extract_po_number(query)
    if not po_no:
        return "Please specify a valid PO number."
    
    df = _df()  # Automatically filters by consignee
    po_col = "po_number_multiple" if "po_number_multiple" in df.columns else "po_number"
    rows = df[df[po_col].astype(str).str.contains(po_no, case=False, na=False)]
    
    if rows.empty:
        return f"No data found for PO {po_no} or you are not authorized to access this PO."
    
    row = rows.iloc[0]
    
    # Check if already arrived
    if pd.notnull(row.get('delivery_date_to_consignee')):
        arrival_date = row['delivery_date_to_consignee']
        return f"PO {po_no} already arrived on {arrival_date.strftime('%Y-%m-%d')}"
    
    # Get best ETA
    eta_fd = row.get('predictive_eta_fd') or row.get('revised_eta_fd') or row.get('eta_fd')
    
    if pd.isna(eta_fd):
        return f"No ETA available for PO {po_no}."
    
    # Get last day of current month
    today = pd.Timestamp.today()
    last_day_of_month = pd.Timestamp(today.year, today.month, 1) + pd.DateOffset(months=1) - pd.Timedelta(days=1)
    
    if eta_fd <= last_day_of_month:
        return f"Yes, PO {po_no} is expected to arrive by {eta_fd.strftime('%Y-%m-%d')} (before month end: {last_day_of_month.strftime('%Y-%m-%d')})"
    else:
        return f"No, PO {po_no} is expected to arrive on {eta_fd.strftime('%Y-%m-%d')} (after month end: {last_day_of_month.strftime('%Y-%m-%d')})"

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
            ("<strong>Departed From</strong>", row.get("load_port"), row.get("atd_lp")),
            ("<strong>Arrived at Final Load Port</strong>", row.get("final_load_port"), row.get("ata_flp")),
            ("<strong>Departed from Final Load Port</strong>", row.get("final_load_port"), row.get("atd_flp")),
            ("<strong>Expected at Discharge Port</strong>", row.get("discharge_port"), row.get("derived_ata_dp")),
            ("<strong>Reached at Discharge Port</strong>", row.get("discharge_port"), row.get("ata_dp")),
            ("<strong>Reached at Last CY</strong>", row.get("last_cy_location"), row.get("equipment_arrived_at_last_cy")),
            ("<strong>Out Gate at Last CY</strong>", row.get("out_gate_at_last_cy_lcn"), row.get("out_gate_at_last_cy")),
            ("<strong>Delivered at</strong>", row.get("delivery_date_to_consignee_lcn"), row.get("delivery_date_to_consignee")),
            ("<strong>Empty Container Returned to</strong>", row.get("empty_container_return_lcn"), row.get("empty_container_return_date")),
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


# def get_delayed_containers(query: str) -> str:
#     """Find containers delayed by specified number of days - supports exact or range queries"""
#     import re
#     df = _df()  # This now automatically filters by consignee
#     df = ensure_datetime(df, ["eta_dp", "ata_dp"])
    
#     if "delay_days" not in df.columns:
#         df["delay_days"] = (df["ata_dp"] - df["eta_dp"]).dt.days
#         df["delay_days"] = df["delay_days"].fillna(0).astype(int)

#     # Enhanced regex to detect different query patterns
#     exact_match = re.search(r"delayed by (?:exactly )?(\d+) days?", query, re.IGNORECASE)
#     more_than_match = re.search(r"delayed by more than (\d+) days?", query, re.IGNORECASE)
#     at_least_match = re.search(r"delayed by at least (\d+) days?", query, re.IGNORECASE)
    
#     if more_than_match or at_least_match:
#         # Range query: more than X days
#         days = int((more_than_match or at_least_match).group(1))
#         delayed_df = df[(df["delay_days"] > days) & (df["delay_days"] > 0)]
#         query_type = f"more than {days}"
#     elif exact_match:
#         # Exact query: exactly X days
#         days = int(exact_match.group(1))
#         delayed_df = df[(df["delay_days"] == days) & (df["delay_days"] > 0)]
#         query_type = f"exactly {days}"
#     else:
#         # Default to exact match for simple "delayed by X days"
#         days_match = re.search(r"(\d+) days?", query, re.IGNORECASE)
#         days = int(days_match.group(1)) if days_match else 7
#         delayed_df = df[(df["delay_days"] == days) & (df["delay_days"] > 0)]
#         query_type = f"exactly {days}"
    
#     if delayed_df.empty:
#         return f"No containers are delayed by {query_type} days for your authorized consignees."
    
#     cols = ["container_number", "eta_dp", "ata_dp", "delay_days"]
#     if "consignee_code_multiple" in delayed_df.columns:
#         cols.append("consignee_code_multiple")
    
#     cols = [c for c in cols if c in delayed_df.columns]
#     delayed_df = delayed_df[cols].sort_values("delay_days", ascending=False)
#     delayed_df = ensure_datetime(delayed_df, ["eta_dp", "ata_dp"])
#     delayed_df["eta_dp"] = delayed_df["eta_dp"].dt.strftime("%Y-%m-%d")
#     delayed_df["ata_dp"] = delayed_df["ata_dp"].dt.strftime("%Y-%m-%d")
    
#     # Return both message and data
#     result_data = delayed_df.to_dict(orient="records")
    
#     message = f"Found {len(result_data)} containers delayed by {query_type} days for your consignee.\n\n"
    
#     # Add examples to the message
#     example_count = min(5, len(result_data))
#     for i, container in enumerate(result_data[:example_count]):
#         delay_days = container['delay_days']
#         message += f"• {container['container_number']}: Expected {container['eta_dp']}, Arrived {container['ata_dp']} ({delay_days} days late)\n"
    
#     if len(result_data) > example_count:
#         message += f"... and {len(result_data) - example_count} more containers.\n"
    
#     # Embed the data for API extraction
#     message += f"\n{result_data}"
    
#     return message

## Get delayed containers



def get_delayed_containers(query: str) -> str:
    """Find containers delayed by specified number of days - supports exact or range queries."""
    import re
 
    df = _df()  # already consignee-filtered if set
    df = ensure_datetime(df, ["eta_dp", "ata_dp"])
 
    # Only arrived containers can be delayed
    arrived = df[df["ata_dp"].notna()].copy()
    if arrived.empty:
        return "No containers have arrived for your authorized consignees."
 
    # Calculate delay days (integer)
    arrived["delay_days"] = (arrived["ata_dp"] - arrived["eta_dp"]).dt.days
    arrived["delay_days"] = arrived["delay_days"].fillna(0).astype(int)
 
    # ---------- STRICT location filter (code or name) ----------
    # Ports to search (priority)
    port_cols = [c for c in ["discharge_port", "vehicle_arrival_lcn", "final_destination", "place_of_delivery"] if c in arrived.columns]
 
    def _extract_loc_code_and_name(q: str):
        q_up = (q or "").upper()
        # 1) explicit code inside parentheses, e.g., (NLRTM)
        m = re.search(r"\(([A-Z0-9]{3,6})\)", q_up)
        if m:
            return m.group(1), None
        # 2) bare code tokens (3–6 letters) – pick those that exist in dataset values
        cand_codes = set(re.findall(r"\b[A-Z0-9]{3,6}\b", q_up))
        if port_cols and cand_codes:
            known_codes = set()
            for c in port_cols:
                vals = arrived[c].dropna().astype(str).str.upper()
                known_codes |= set(re.findall(r"\(([A-Z0-9]{3,6})\)", " ".join(vals.tolist())))
            for code in cand_codes:
                if code in known_codes:
                    return code, None
        # 3) fallback port name phrase after 'at|in|to'
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
            # name match: require presence of each 3+ char token
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
 
    # Patterns (now include 'late/overdue/behind' synonyms)
    more_than_match   = re.search(
        r"(?:(?:delayed|late|overdue|behind)\s+by\s+)?(?:more than|over|>\s*)\s*(\d+)\s*days?",
        q, re.IGNORECASE
    )
    at_least_explicit = re.search(
        r"(?:(?:delayed|late|overdue|behind)\s+by\s+)?(?:at\s+least|>=\s*|greater\s+than\s+or\s+equal\s+to|minimum)\s*(\d+)\s*days?",
        q, re.IGNORECASE
    )
    or_more_match     = re.search(r"\b(\d+)\s+days?\s*(?:or\s+more|and\s+more|or\s+above)\b", q, re.IGNORECASE)
    exact_phrase_match= re.search(r"(?:delayed|late|overdue|behind)\s+by\s+(\d+)\s+days?", q, re.IGNORECASE)
    plain_days_match  = re.search(r"\b(\d+)\s+days?\b(?:\s*(?:late|delayed|overdue|behind))?", q, re.IGNORECASE)
 
    # Decide filter semantics
    if more_than_match:
        days = int(more_than_match.group(1))
        delayed_df = arrived[arrived["delay_days"] > days]
        query_type = f"more than {days}"
    elif at_least_explicit or or_more_match:
        days = int((at_least_explicit or or_more_match).group(1))
        delayed_df = arrived[arrived["delay_days"] >= days]
        query_type = f"at least {days}"
    elif exact_phrase_match or plain_days_match:
        # Treat "5 days", "delayed by 5 days", or "5 days late" as exactly X
        days = int((exact_phrase_match or plain_days_match).group(1))
        delayed_df = arrived[arrived["delay_days"] == days]
        query_type = f"exactly {days}"
    else:
        # Fallback: any positive delay
        delayed_df = arrived[arrived["delay_days"] > 0]
        query_type = "more than 0"
 
    # Only positive delays
    delayed_df = delayed_df[delayed_df["delay_days"] > 0]
 
    if delayed_df.empty:
        where = f" at {code or name}" if (code or name) else ""
        return f"No containers are delayed by {query_type} days for your authorized consignees{where}."
 
    cols = ["container_number", "eta_dp", "ata_dp", "delay_days",
            "consignee_code_multiple", "discharge_port"]
    if "vehicle_arrival_lcn" in delayed_df.columns:
        cols.append("vehicle_arrival_lcn")
    cols = [c for c in cols if c in delayed_df.columns]
    out = delayed_df[cols].sort_values("delay_days", ascending=False).copy()
 
    # Format dates
    if "eta_dp" in out.columns and pd.api.types.is_datetime64_any_dtype(out["eta_dp"]):
        out["eta_dp"] = out["eta_dp"].dt.strftime("%Y-%m-%d")
    if "ata_dp" in out.columns and pd.api.types.is_datetime64_any_dtype(out["ata_dp"]):
        out["ata_dp"] = out["ata_dp"].dt.strftime("%Y-%m-%d")
 
    return out.to_dict(orient="records")




# ------------------------------------------------------------------
# 3️⃣ Upcoming Arrivals (next X days)
# ------------------------------------------------------------------
def get_upcoming_arrivals(query: str) -> str:
    """
    List containers scheduled to arrive within the next X days.
    Now filters by authorized consignee codes.
    """
    # Enhanced regex patterns to catch various formats
    patterns = [
        r"(?:next|upcoming|in)\s+(\d+)\s+days?",  # "next 2 days", "in 2 days"
        r"(\d+)\s+days?",                         # "2 days"
        r"arriving.*?(\d+)\s+days?",              # "arriving in 2 days"
        r"will.*?arrive.*?(\d+)\s+days?",         # "will arrive in 2 days"
    ]
    
    days = None
    for pattern in patterns:
        m = re.search(pattern, query, re.IGNORECASE)
        if m:
            days = int(m.group(1))
            break
    
    # Default to 7 days if no number found
    if days is None:
        days = 7

    # Use filtered DataFrame
    df = _df()
    eta_col = next((c for c in ["eta_dp", "eta", "estimated_time_arrival"] if c in df.columns), None)
    if not eta_col:
        return "ETA column not found in the data."

    df = ensure_datetime(df, [eta_col])
    today = pd.Timestamp.today().normalize()
    future = today + pd.Timedelta(days=days)

    # Filter by ETA window and null ATA (not yet arrived)
    upcoming = df[
        (df[eta_col] >= today) & 
        (df[eta_col] <= future) & 
        (df.get('ata_dp', pd.Series([pd.NaT] * len(df))).isna())  # Not yet arrived
    ]
    
    if upcoming.empty:
        return f"No containers scheduled to arrive in the next {days} days for your authorized consignees."

    # Include consignee column in the display
    cols = ["container_number", "discharge_port", eta_col]
    if "consignee_code_multiple" in upcoming.columns:
        cols.append("consignee_code_multiple")
    
    cols = [c for c in cols if c in upcoming.columns]
    
    upcoming = upcoming[cols].sort_values(eta_col)
    upcoming = ensure_datetime(upcoming, [eta_col])
    upcoming[eta_col] = upcoming[eta_col].dt.strftime("%Y-%m-%d")
    
    # Return as string for better formatting
    result_lines = [f"Containers scheduled to arrive in the next {days} days:"]
    for _, row in upcoming.iterrows():
        consignee = row.get('consignee_code_multiple', 'Unknown')
        result_lines.append(f"- {row['container_number']} at {row['discharge_port']} on {row[eta_col]} (Consignee: {consignee})")
    
    #return "\n".join(result_lines)
    return upcoming.to_dict(orient="records")


# ------------------------------------------------------------------
# 4️⃣ Container ETA/ATA (single container)
# ------------------------------------------------------------------
def get_container_eta(query: str) -> str:
    """
    Return ETA and ATA details for specific containers.
    Input: Query mentioning one or more container numbers (comma-separated or space-separated).
    Output: ETA/ATA and port details for the containers.
    """
    # Extract all container numbers using regex pattern
    container_pattern = re.findall(r'([A-Z]{4}\d{7})', query)
   
    if not container_pattern:
        return "Please mention one or more container numbers."
   
    df = _df()
   
    # Add "MM/dd/yyyy hh:mm:ss tt" format to ensure_datetime function
    # or directly parse dates here
    date_cols = ["eta_dp", "ata_dp"]
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
            cols = ["container_number", "discharge_port", "eta_dp", "ata_dp"]
            cols = [c for c in cols if c in row.index]
            single_result = row[cols].to_frame().T
            results.append(single_result)
        else:
            # Create a row with "Not Available" for missing containers
            missing_row = pd.DataFrame({
                "container_number": [cont],
                "discharge_port": ["Not Available"],
                "eta_dp": ["Not Available"],
                "ata_dp": ["Not Available"]
            })
            results.append(missing_row[["container_number", "discharge_port", "eta_dp", "ata_dp"]])
   
    # Combine all results
    combined_results = pd.concat(results, ignore_index=True)
   
    # Format date columns (only for actual datetime values)
    for date_col in ["eta_dp", "ata_dp"]:
        if date_col in combined_results.columns:
            combined_results[date_col] = combined_results[date_col].apply(
                lambda x: x.strftime("%Y-%m-%d") if isinstance(x, pd.Timestamp) else x
            )
   
    return combined_results.to_string(index=False)


# ------------------------------------------------------------------
# 5️⃣ Arrivals By Port / Country
# ------------------------------------------------------------------
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

    q = query.strip()
    
    # Case 1: "PORT in 10 days" or "PORT in next 10 days"
    q = re.sub(r'(\b[A-Z]{3,6})\s+in\s+(?:next\s+)?(\d{1,3})\s*days?', 
           r'\1 , \2 days', q, flags=re.IGNORECASE)
    
    # Case 2: "PORT 10 days"
    q = re.sub(r'(\b[A-Z]{3,6})\s+(\d{1,3})\s*days?', 
           r'\1 , \2 days', q, flags=re.IGNORECASE)
    
    # Case 3: "10 days PORT"
    q = re.sub(r'(\d{1,3})\s*days?\s+(\b[A-Z]{3,6})', 
           r'\2 , \1 days', q, flags=re.IGNORECASE)
    
    query = q
 
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
def get_containers_arriving_soon(query: str) -> str:
    """
    List containers arriving soon (ETA window, ATA is null) - now with consignee filtering.
    """
    m = re.search(r"next\s+(\d+)\s+days", query, re.IGNORECASE)
    days = int(m.group(1)) if m else 7

    df = _df()  # This now automatically filters by consignee
    df = ensure_datetime(df, ["eta_dp", "ata_dp"])

    today = pd.Timestamp.today().normalize()
    future = today + pd.Timedelta(days=days)

    mask = (df["ata_dp"].isna()) & (df["eta_dp"] >= today) & (df["eta_dp"] <= future)
    subset = df[mask]

    if subset.empty:
        return f"No containers arriving in the next {days} days for your authorized consignees (or they have already arrived)."

    subset = ensure_datetime(subset, ["eta_dp"])
    subset["eta_dp"] = subset["eta_dp"].dt.strftime("%Y-%m-%d")
    
    cols = ["container_number", "eta_dp", "discharge_port", "consignee_code_multiple"]
    cols = [c for c in cols if c in subset.columns]
    
    return subset[cols].head(10).to_string(index=False)


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

    df = _df()  # This now automatically filters by consignee
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
        return "Please provide a valid container number (e.g., TCLU8579495) or PO number to check arrival status."


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
        return "Please specify a PO number (e.g. 'PO 5500009022' or '5500009022')."
 
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
        return "Please specify a PO number (e.g. 'PO 5500009022' or '5500009022')."
 
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
    Loads the shipment CSV from Azure Blob and creates a persistent SQLite engine for SQL queries.
    """
    from agents.azure_agent import get_persistent_sql_engine
    return get_persistent_sql_engine()


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
        description="Find containers delayed by a specified number of days."
    ),
    Tool(
        name="Get Upcoming Arrivals",
        func=get_upcoming_arrivals,
        description="List containers scheduled to arrive within the next X days."
    ),
    Tool(
        name="Get Container ETA",
        func=get_container_eta,
        description="Return ETA and ATA details for a specific container."
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
        name="Get Hot Containers",
        func=get_hot_containers,
        description="Get list of hot containers for authorized consignees based on hot container flag"
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
        name="Get Containers By Transport Mode",
        func=get_containers_by_transport_mode,
        description="Find containers filtered by transport_mode (e.g. 'arrived by sea', 'arrive by air in next 3 days')."
    ),
    Tool(
        name="Handle Non-shipping queries",
        func=handle_non_shipping_queries,
        description="This is for non-shipping generic queries. Like 'how are you' or 'hello' or 'hey' or 'who are you' etc."
    )
    
]
































































