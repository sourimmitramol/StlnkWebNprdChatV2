from agents.prompts import map_synonym_to_column
from agents.tools import (
    sql_query_tool,
    get_container_milestones,
    get_container_status,
    get_container_carrier,
    check_arrival_status,
    get_delayed_containers,
    get_upcoming_arrivals,
    get_container_eta,
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
    check_po_month_arrival,
    get_weekly_status_changes,
    get_current_location,
)

def route_query(query: str, consignee_codes: list = None) -> str:
    """Route query with consignee authorization support"""
    # Set thread-local consignee codes for tools
    if consignee_codes:
        import threading
        threading.current_thread().consignee_codes = consignee_codes
    
    q = query.lower()
    
    # Hot containers routing
    if "hot container" in q or "hot containers" in q:
        from agents.tools import get_hot_containers_by_consignee
        return get_hot_containers_by_consignee(query)
    
    # Enhanced container/PO/OBL/Booking detection
    from utils.container import extract_container_number, extract_po_number, extract_ocean_bl_number, extract_carrier_booking_number
    
    container_no = extract_container_number(query)
    po_no = extract_po_number(query) 
    bl_no = extract_ocean_bl_number(query)
    booking_no = extract_carrier_booking_number(query)
    
    # Question 7: String could be container, PO, OBL, or booking
    if any([container_no, po_no, bl_no, booking_no]) and not any(keyword in q for keyword in ["carrier", "status", "milestone", "arrived"]):
        return get_field_info(query)
    
    # Question 8: POs shipping in coming days
    if "po" in q and ("ship" in q or "etd" in q) and ("coming" in q or "next" in q):
        return get_upcoming_pos(query)
    
    # Question 14: Cargo in transit
    if ("cargo" in q or "po" in q) and "transit" in q:
        return check_transit_status(query)
    
    # Question 19-22: Carrier/Supplier based queries
    if "carrier" in q and ("container" in q or "ship" in q) and ("last" in q or "days" in q):
        return get_containers_by_carrier(query)
    
    if "supplier" in q and ("container" in q or "po" in q):
        return get_containers_by_supplier(query)
    
    # Question 24: PO arrival by month end
    if "po" in q and ("arrive" in q or "destination" in q) and ("month" in q or "end" in q):
        return check_po_month_arrival(query)
    
    # Question 25: Delayed POs
    if "po" in q and ("delay" in q or "delaying" in q):
        return get_delayed_pos(query)
    
    # Question 27: Weekly status changes
    if "status" in q and ("week" in q or "change" in q):
        return get_weekly_status_changes(query)
    
    # Question 30: Current location
    if "where" in q and ("container" in q or "location" in q or "right now" in q):
        return get_current_location(query)
    
    # Existing routing logic...
    # ... rest of your current routing logic
    # Add carrier routing - check for carrier keywords
    if ("carrier" in q or "shipping line" in q) and ("container" in q or "po" in q or any(char.isdigit() for char in q)):
        return get_container_carrier(query)
    elif ("arrived" in q or "reached" in q or "arrival" in q or "on water" in q or "on the water" in q) and ("container" in q or "po" in q or any(char.isdigit() for char in q)):
        return check_arrival_status(query)
    elif "status" in q and ("container" in q or any(char.isdigit() for char in q)):
        return get_container_status(query)
    elif "milestone or status" in q:
        return get_container_milestones(query)
    elif "milestone" in q and ("container" in q or any(char.isdigit() for char in q)):
        return get_container_milestones(query)
    elif "delay" in q:
        return get_delayed_containers(query)
    elif "upcoming arrival" in q or "arriving soon" in q:
        return get_upcoming_arrivals(query)
    elif "eta" in q or "ata" in q:
        return get_container_eta(query)
    elif "port" in q:
        return get_arrivals_by_port(query)
    elif "keyword" in q or "search" in q:
        return lookup_keyword(query)
    elif "analyze" in q or "average" in q or "total" in q:
        return analyze_data_with_pandas(query)
    elif "field info" in q:
        return get_field_info(query)
    elif "vessel" in q:
        return get_vessel_info(query)
    elif "po" in q and "upcoming" in q:
        return get_upcoming_pos(query)
    elif "po" in q and "delay" in q:
        return get_delayed_pos(query)
    elif "container" in q and "arriving soon" in q:
        return get_containers_arriving_soon(query)
    elif "lp" in q or "load port" in q:
        return get_load_port_for_container(query)
    elif "semantic" in q or "vector" in q or "similar" in q:
        return vector_search_tool(query)
    elif "sql" in q or "database" in q or "table" in q:
        return sql_query_tool(query)
    elif "column" in q or "mapping" in q or "synonym" in q:
        return answer_with_column_mapping(query)
    else:
        # Fallback: Provide a more detailed default response
        return (
            "Thought: The query did not match any specialized tool or known pattern.\n"
            "Final Answer: Sorry, I couldn't understand your query. Please rephrase or provide more details about the container, port, milestone, or shipment event you are interested in."
        )
    
    # Clear consignee codes after routing
    if consignee_codes:
        import threading
        if hasattr(threading.current_thread(), 'consignee_codes'):
            delattr(threading.current_thread(), 'consignee_codes')

# Usage example in your FastAPI endpoint:
# from agents.router import route_query
# result = route_query(user_query)




