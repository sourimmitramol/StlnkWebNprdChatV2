from agents.prompts import map_synonym_to_column
from agents.tools import (
    sql_query_tool,
    get_container_milestones,
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
    check_arrival_status,
    get_container_carrier,
    answer_with_column_mapping,
    vector_search_tool,
    get_blob_sql_engine,
    #  # Make sure this is defined in tools.py
)

def route_query(query: str) -> str:
    q = query.lower()
    """provide a response in the tabular form"""
    if any("milestone", "status", "track", "event history", "journey", "where") in q:
        return get_container_milestones(query)
    elif ("carrier" in q or "shipping line" in q) and ("container" in q or "po" in q or any(char.isdigit() for char in q)):
        return get_container_carrier(query)
    elif ("arrived" in q or "reached" in q or "arrival" in q or "on water" in q or "on the water" in q) and ("container" in q or "po" in q or any(char.isdigit() for char in q)):
        return check_arrival_status(query)
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

# Usage example in your FastAPI endpoint:
# from agents.router import route_query

# result = route_query(user_query)






