import re
import threading
from utils.logger import logger
from agents.prompts import map_synonym_to_column
from agents.tools import (
    sql_query_tool,
    get_container_milestones,
    get_container_carrier,
    check_arrival_status,
    get_delayed_containers,
    get_upcoming_arrivals,
    get_container_etd,
    get_arrivals_by_port,
    lookup_keyword,
    analyze_data_with_pandas,
    get_field_info,
    get_vessel_info,
    get_upcoming_pos,
    get_delayed_pos,
    get_containers_arriving_soon,
    get_load_port_for_container,
    answer_with_column_mapping,
    vector_search_tool,
    get_blob_sql_engine,
    # Add the missing functions
    check_transit_status,
    get_containers_by_carrier,
    get_containers_by_supplier,
    get_hot_upcoming_arrivals,
    check_po_month_arrival,
    get_weekly_status_changes,
    get_hot_containers,  # Add this missing import
    _df,  # Import the DataFrame function to test filtering
)

def validate_consignee_filtering(consignee_codes: list) -> bool:
    """Validate that consignee filtering is working properly"""
    try:
        # Get filtered DataFrame
        df_filtered = _df()
        
        if df_filtered.empty:
            logger.warning(f"No data found after consignee filtering for codes: {consignee_codes}")
            return False
        
        # Check if consignee codes are present in the filtered data
        if 'consignee_code_multiple' in df_filtered.columns:
            # Extract numeric codes from consignee_codes
            import re
            numeric_codes = []
            for code in consignee_codes:
                # Extract numeric part (e.g., "0045831" from "EDDIE BAUER LLC(0045831)")
                match = re.search(r'\((\d+)\)', code)
                if match:
                    numeric_codes.append(match.group(1))
                else:
                    # If already just numeric, use as is
                    numeric_codes.append(code.strip())
            
            # Check if any of the codes appear in the filtered data
            pattern = r"|".join([rf"\b{re.escape(code)}\b" for code in numeric_codes])
            has_authorized_data = df_filtered['consignee_code_multiple'].astype(str).apply(
                lambda x: bool(re.search(pattern, x))
            ).any()
            
            if not has_authorized_data:
                logger.warning(f"Filtered data doesn't contain authorized consignee codes: {numeric_codes}")
                return False
            
            logger.info(f"Consignee filtering validated: {len(df_filtered)} rows for codes {numeric_codes}")
            return True
        else:
            logger.warning("No consignee_code_multiple column found in filtered data")
            return False
            
    except Exception as e:
        logger.error(f"Error validating consignee filtering: {e}", exc_info=True)
        return False

def route_query(query: str, consignee_codes: list = None) -> str:
    """Route query with consignee authorization support"""
    try:
        # Set thread-local consignee codes for tools
        if consignee_codes:
            threading.current_thread().consignee_codes = consignee_codes
            logger.info(f"Router: Set consignee codes {consignee_codes} for query: {query}")
            
            # ========== VALIDATE CONSIGNEE FILTERING ==========
            if not validate_consignee_filtering(consignee_codes):
                return f"No data available for your authorized consignee codes: {consignee_codes}"
        
        q = query.lower()
        
        # Enhanced container/PO/OBL/Booking detection (do this early)
        from utils.container import extract_container_number, extract_po_number
        container_no = extract_container_number(query)
        po_no = extract_po_number(query) 

        # Raju's statement starts from here...
        # ========== PRIORITY 0: SQL Queries for complex analytical questions ==========
        sql_trigger_phrases = [
            "how many", "count of", "list all", "show me", "display", "get me",
            "which", "what is the", "find all", "statistics", "analytics",
            "average", "total", "sum of", "maximum", "minimum", "highest", "lowest",
            "group by", "order by", "sort by", "filter by", "where", "select",
            "group", "aggregate", "categorize", "break down by", "by mode", "by type"
        ]
 
        sql_exclude_phrases = [
            "status of", "track container", "milestone", "carrier for po",
            "carrier for container", "supplier for", "hot container status",
            "delay status", "where is container", "current location"
        ]
       
 
        # Check if query should use SQL tool
        use_sql_tool = (
            any(phrase in q for phrase in sql_trigger_phrases) and
            not any(exclude in q for exclude in sql_exclude_phrases)
        )
 
        # Also use SQL for general data exploration queries
        if any(explore_phrase in q for explore_phrase in [
            "containers at", "containers from", "containers in", "containers arriving",
            "containers coming", "shipments from", "shipments to", "data for", "records of",
            "group by", "categorize", "breakdown"
        ]):
            use_sql_tool = True
 
        # FORCE SQL for analytical operations
        if any(force_sql in q for force_sql in ["select", "show me", "get me", "group by", "count", "sum", "average", "aggregate"]):
            use_sql_tool = True
 
        if use_sql_tool:
            try:
                result = sql_query_tool(query)
                # If SQL tool returns a valid response, use it
                if result and "No matching data" not in result and "failed" not in result.lower():
                    logger.info(f"Result: {result}")
                    return result
                # If SQL fails, continue to specific tools
            except Exception as e:
                logger.warning(f"SQL tool failed, falling back: {e}")              
        # my statement ends here...
        
        
        # ========== PRIORITY 1: Handle port/location queries ==========
        if any(phrase in q for phrase in ["list containers from", "containers from", "from port"]):
            return get_arrivals_by_port(query)
        
        # ========== PRIORITY 2: Handle delay queries ==========
        if ("delay" in q or "late" in q or "overdue" in q or "missed" in q) and "days" in q:
            return get_delayed_containers(query)
        
        # ========== PRIORITY 3: Handle carrier queries (NEW - SPECIFIC) ==========
        # Questions 15, 16, 17, 18: Carrier queries for PO/Container/OBL
        if ("carrier" in q or "who is" in q) and (po_no or container_no or "obl" in q):
            return get_container_carrier(query)
        
        # ========== PRIORITY 4: Hot containers routing ==========
        if "hot container" in q or "hot containers" in q:
            return get_hot_containers(query)
        
        # ========== PRIORITY 5: Container status queries ==========
        if any(keyword in q for keyword in ["milestone", "status", "track", "event history", "journey", "where"]):
            return get_container_milestones(query)
        
        # ========== PRIORITY 6: Question 8 - POs shipping in coming days ==========
        elif "po" in q and ("ship" in q or "etd" in q) and ("coming" in q or "next" in q):
            return get_upcoming_pos(query)
        
        # ========== PRIORITY 7: Question 14 - Cargo in transit ==========
        if ("cargo" in q or "po" in q) and "transit" in q:
            return check_transit_status(query)
        
        # ========== PRIORITY 8: Question 19-22 - Carrier/Supplier based queries ==========
        if "carrier" in q and ("container" in q or "ship" in q) and ("last" in q or "days" in q):
            return get_containers_by_carrier(query)
        
        if "supplier" in q and ("container" in q or "po" in q):
            return get_containers_by_supplier(query)
        
        # ========== PRIORITY 9: Question 24 - PO arrival by month end ==========
        if "po" in q and ("arrive" in q or "destination" in q) and ("month" in q or "end" in q):
            return check_po_month_arrival(query)
        
        # ========== PRIORITY 10: Question 25 - Delayed POs ==========
        if "po" in q and ("delay" in q or "delaying" in q):
            return get_delayed_pos(query)
        
        # ========== PRIORITY 11: Question 27 - Weekly status changes ==========
        if "status" in q and ("week" in q or "change" in q):
            return get_weekly_status_changes(query)       
        
        # ========== PRIORITY 13: Upcoming arrivals ==========
        if ("arriving" in q or "arrive" in q) and ("next" in q or "coming" in q):
            return get_upcoming_arrivals(query)
        
        # ========== PRIORITY 14: Question 7 - General field info (MOVED TO END) ==========
        # This is now more specific and won't interfere with carrier queries
        if any([container_no, po_no]) and not any(keyword in q for keyword in [
            "carrier", "status", "milestone", "arrived","track", "delay", "ship", "transit", 
            "supplier", "where", "location","event history", "journey"
        ]):
            return get_field_info(query)
        
        # Default fallback
        return "I couldn't understand your query. Please try rephrasing or provide more specific information."
        
    except Exception as e:
        logger.error(f"Error in route_query: {e}", exc_info=True)
        return f"Error processing your query: {str(e)}"
    finally:
        # ========== ALWAYS CLEAN UP CONSIGNEE CONTEXT ==========
        if consignee_codes and hasattr(threading.current_thread(), 'consignee_codes'):
            delattr(threading.current_thread(), 'consignee_codes')
            logger.debug("Cleaned up consignee codes from thread context")














