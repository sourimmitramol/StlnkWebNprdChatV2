# api/main.py
import logging
import re
import ast
from fastapi import FastAPI, HTTPException, Query, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List

from utils.container import extract_po_number

from .schemas import AskRequest, QueryWithConsigneeBody
from agents.azure_agent import initialize_azure_agent
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
    get_weekly_status_changes,
    get_hot_containers,  # Add this
    _df
)
from agents.router import route_query

logger = logging.getLogger("shipping_chatbot")

app = FastAPI(
    title="Shipping Chatbot API",
    version="1.0.0",
    description="FastAPI wrapper around the Azure‑OpenAI agent you built."
)

# ----------------------------------------------------------------------
# CORS – allow any origin (adjust for production!)
# ----------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# Startup – create the agent once
# ----------------------------------------------------------------------
@app.on_event("startup")
def startup():
    global AGENT, LLM
    AGENT, LLM = initialize_azure_agent()  # imports all tools automatically
    logger.info("Agent ready for HTTP requests")


@app.get("/health")
def health():
    """Simple health check."""
    return {"status": "ok", "message": "Shipping chatbot is alive"}


# ----------------------------------------------------------------------
# Helper function to validate consignee access
# ----------------------------------------------------------------------
def filter_by_consignee(df, consignee_codes: List[str]):
    """Filter DataFrame to only include rows where consignee_code_multiple contains any of the provided codes"""
    pattern = r"|".join([rf"\b{re.escape(code)}\b" for code in consignee_codes])
    mask = df['consignee_code_multiple'].astype(str).apply(
        lambda x: bool(re.search(pattern, x))
    )
    return df[mask]


def sanitize_response(response_text: str) -> str:
    """
    Remove consignee code references from response text to avoid exposing internal codes to users.
    
    Removes patterns like:
    - "for user 0000866"
    - "for consignee 0000866"
    - "for consignee code 0000866"
    - "user 0000866,0001363"
    - "for user code 0000866"
    
    Args:
        response_text: The response text to sanitize
        
    Returns:
        Sanitized response text without consignee code references
    """
    if not response_text or not isinstance(response_text, str):
        return response_text
    
    # Pattern 1: Remove "for user XXXXX" or "for consignee XXXXX" (single or comma-separated codes)
    response_text = re.sub(
        r'\s+for\s+(?:user|consignee)(?:\s+code[s]?)?\s+[\d,\s]+',
        '',
        response_text,
        flags=re.IGNORECASE
    )
    
    # Pattern 2: Remove "user XXXXX:" at the beginning (e.g., "user 0000866: What are...")
    response_text = re.sub(
        r'^\s*(?:user|consignee)(?:\s+code[s]?)?\s+[\d,\s]+\s*:\s*',
        '',
        response_text,
        flags=re.IGNORECASE
    )
    
    # Pattern 3: Remove standalone "for user/consignee code(s) XXXXX"
    response_text = re.sub(
        r'\s+for\s+(?:the\s+)?(?:user|consignee)(?:\s+code[s]?)?\s+[\d,\s]+',
        '',
        response_text,
        flags=re.IGNORECASE
    )
    
    # Pattern 4: Clean up any double spaces or trailing/leading whitespace
    response_text = re.sub(r'\s+', ' ', response_text).strip()
    
    return response_text


def check_container_authorization(df, container_numbers: List[str], consignee_codes: List[str]):
    """Check if the container belongs to one of the authorized consignees"""
    authorized_df = filter_by_consignee(df, consignee_codes)
    container_mask = authorized_df['container_number'].isin(container_numbers)
    return authorized_df[container_mask], not authorized_df[container_mask].empty

# ...existing code...


@app.post("/ask")
def ask(body: QueryWithConsigneeBody):
    q = body.question.strip()
    #consignee_codes = [c.strip() for c in body.consignee_code.split(",") if c.strip()]
    consignee_codes = list(dict.fromkeys(c.strip() for c in body.consignee_code.split(",") if c.strip()))
    #print(consignee_codes)
    logger.info(f"User_Query: {q}, c_codes: {consignee_codes}")
    if not q:
        raise HTTPException(status_code=400, detail="Empty question")
    
    if not consignee_codes:
        raise HTTPException(status_code=400, detail="Consignee code is required")

    # Get the DataFrame from blob storage
    from services.azure_blob import get_shipment_df
    df = get_shipment_df().copy()
    
    # Filter by consignee codes
    authorized_df = filter_by_consignee(df, consignee_codes)
    
    if authorized_df.empty:
        return {"response": "No data found.", "table": [], "mode": "agent"}

    # Extract container numbers from the query if present
    container_pattern = r'(?:container(?:s)?\s+(?:number(?:s)?)?(?:\s+is|\s+are)?\s+)?([A-Z]{4}\d{7}(?:\s*,\s*[A-Z]{4}\d{7})*)'
    container_match = re.search(container_pattern, q, re.IGNORECASE)
    
    if container_match:
        # Extract container numbers
        container_str = container_match.group(1)
        requested_containers = [c.strip() for c in re.split(r'\s*,\s*', container_str)]
        
        # Check if these containers belong to the authorized consignees
        _, is_authorized = check_container_authorization(df, requested_containers, consignee_codes)
        if not is_authorized:
            return {"response": "No data found.", "table": [], "mode": "agent"}
    
    # ---------- NEW: helper to pull milestone Observation ----------
    # Include actual tool names and normalize to lowercase for matching.
    MILESTONE_TOOL_NAMES = {n.lower() for n in [
        "Get Container Milestones",
        "get_container_milestones",
        "Milestone Lookup",
        "Check Arrival Status",
        "check_arrival_status",
        "Get Container Status",
    ]}
    def _extract_milestone_observation(intermediate_steps):
        """
        Return the latest observation text from milestone/status tools, if any.
        Supports LC tuples (AgentAction, observation) and dict payloads.
        """
        if not intermediate_steps:
            return None
        obs = None
        for step in intermediate_steps:
            if isinstance(step, tuple) and len(step) == 2:
                action, observation = step
                tool = getattr(action, "tool", None) or getattr(action, "tool_name", None)
            elif isinstance(step, dict):
                action = step.get("action")
                observation = step.get("observation")
                tool = None
                if action is not None:
                    tool = getattr(action, "tool", None)
                    if tool is None and isinstance(action, dict):
                        tool = action.get("tool")
            else:
                continue

            tool_name = ((tool or "").strip()).lower()
            if tool_name in MILESTONE_TOOL_NAMES or ("milestone" in tool_name):
                obs = observation  # keep last
        return obs
    # ---------------------------------------------------------------

    try:
        # Pass consignee context to the agent including authorization info
        consignee_context = f"user {','.join(consignee_codes)}: {q}"
        #consignee_context = f"user: {q}"

        # Set consignee codes in a global context that tools can access
        import threading
        threading.current_thread().consignee_codes = consignee_codes

        try:
            result = AGENT.invoke({"input": consignee_context})
        except TypeError:
            result = AGENT.invoke({"input": consignee_context})

        # Clear the context after use
        if hasattr(threading.current_thread(), 'consignee_codes'):
            delattr(threading.current_thread(), 'consignee_codes')

        if isinstance(result, str):
            result = {"output": result, "intermediate_steps": []}

        output = (result.get("output") or "").strip()
        print(result.get("intermediate_steps"))
        milestone_obs = _extract_milestone_observation(result.get("intermediate_steps", []))

        # Fallback: try to extract inline "Observation: ... [Final Answer: ...]" from the LLM output
        inline_obs = None
        try:
            m = re.search(r"(Observation:\s*.+?)(?:\n\s*Final Answer:|\Z)", output, flags=re.DOTALL | re.IGNORECASE)
            if m:
                inline_obs = m.group(1).strip()
        except Exception:
            pass

        # ---- NEW: Fallback when agent stops due to iteration/time limit ----
        if re.search(r"Agent stopped due to iteration limit or time limit\.", output, re.IGNORECASE):
            fallback = route_query(q, consignee_codes)  # Pass consignee codes
            
            # Handle list/dict returns from transit analysis functions
            if isinstance(fallback, list) and len(fallback) > 0 and isinstance(fallback[0], dict):
                # Transit analysis functions return [{"summary": {...}, "container_details": [...}}]
                item = fallback[0]
                if "summary" in item and "container_details" in item:
                    summary = item["summary"]
                    containers = item["container_details"]
                    
                    # Format summary message
                    if "ocean_bl_number" in summary:
                        msg = f"Transit Analysis for Ocean BL: {summary['ocean_bl_number']}\n\n"
                    elif "po_number" in summary:
                        msg = f"Transit Analysis for PO: {summary['po_number']}\n\n"
                    else:
                        msg = "Transit Analysis Summary:\n\n"
                    
                    msg += f"Total Containers: {summary.get('total_containers', 0)}\n"
                    msg += f"Arrived: {summary.get('arrived_containers', 0)}, In Transit: {summary.get('in_transit_containers', 0)}\n"
                    msg += f"Average Transit Time: {summary.get('avg_transit_days', 'N/A')} days\n"
                    msg += f"Average Delay: {summary.get('avg_delay_days', 'N/A')} days\n"
                    msg += f"Delayed Containers: {summary.get('delayed_containers', 0)}\n"
                    msg += f"On-Time/Early: {summary.get('on_time_or_early_containers', 0)}"
                    
                    return {
                        "response": sanitize_response(msg),
                        "observation": sanitize_response(msg),
                        "table": containers,  # Container details as table
                        "mode": "router-fallback"
                    }
            
            # Default string response
            fallback_str = str(fallback) if not isinstance(fallback, str) else fallback
            return {
                "response": sanitize_response(fallback_str),
                "observation": sanitize_response(fallback_str),
                "table": [],
                "mode": "router-fallback"
            }
        # --------------------------------------------------------------------

        # -------- existing table extraction logic (operates on `output` only) ---
        table_data = []
        json_match = re.search(r"(\[{.*?}\])", output, re.DOTALL)
        if json_match:
            try:
                json_text = json_match.group(1)
                table_data = ast.literal_eval(json_text)
                output = output.replace(json_text, "").strip()
                output = re.sub(r'```(json|python)?\s*', '', output)
                output = re.sub(r'\s*```', '', output)
                output = re.sub(r'\[\s*\]', '', output).strip()
            except (SyntaxError, ValueError) as e:
                logger.warning(f"Failed to parse JSON in response: {e}")
                
        if not table_data and "```" in output:
            try:
                code_blocks = re.findall(r'```(?:json|python)?\s*([\s\S]*?)```', output)
                if code_blocks:
                    for block in code_blocks:
                        try:
                            parsed_data = ast.literal_eval(block.strip())
                            if isinstance(parsed_data, list) and parsed_data and isinstance(parsed_data[0], dict):
                                table_data = parsed_data
                                output = output.replace(f"```{block}```", "").strip()
                                break
                        except (SyntaxError, ValueError):
                            continue
            except Exception as e:
                logger.warning(f"Failed to extract table data from markdown: {e}")
        
        if not table_data:
            try:
                for tool_name in ["SQL Query Shipment Data", "Vector Search"]:
                    tool_pattern = fr"{tool_name}:\s*([\s\S]*?)(?:\n\n|$)"
                    tool_match = re.search(tool_pattern, output)
                    if tool_match:
                        tool_output = tool_match.group(1).strip()
                        try:
                            if tool_output.startswith("[") and tool_output.endswith("]"):
                                parsed_tool_data = ast.literal_eval(tool_output)
                                if isinstance(parsed_tool_data, list) and parsed_tool_data:
                                    table_data = parsed_tool_data
                                    output = output.replace(tool_match.group(0), "").strip()
                                    break
                        except (SyntaxError, ValueError):
                            continue
            except Exception as e:
                logger.warning(f"Failed to extract tool output: {e}")
        
        if table_data:
            import pandas as pd
            import numpy as np
            import json
            def clean_non_serializable(obj):
                if isinstance(obj, dict):
                    return {k: clean_non_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_non_serializable(i) for i in obj]
                elif pd.isna(obj) or obj in [float('inf'), float('-inf')]:
                    return None
                elif isinstance(obj, pd.Timestamp):
                    return obj.strftime('%Y-%m-%d')
                else:
                    return obj
            table_data = clean_non_serializable(table_data)
            try:
                json.dumps(table_data)
            except TypeError:
                logger.error("Failed to serialize table data to JSON")
                table_data = []
        # ------------------------------------------------------------------------

        # Prefer the tool observation; fallback to inline observation; else use summary.
        observation_text = None
        if milestone_obs:
            observation_text = str(milestone_obs).strip()
            message = observation_text
        elif inline_obs:
            observation_text = inline_obs
            message = inline_obs
        else:
            message = re.sub(r'\n\s*\n', '\n\n', output).strip() or "No results found for the given query."
            
        observation = result.get("intermediate_steps", [])

        return {
            "response": sanitize_response(message),              # now surfaces the detailed observation in Postman
            "observation": observation,  # explicit field for clients that need it
            "table": table_data,
            "mode": "agent"
        }

    except Exception as exc:
        logger.error(f"Agent failed: {exc}", exc_info=True)
        
        # Try router fallback before failing
        try:
            fallback = route_query(q, consignee_codes)  # Pass consignee codes
            
            # Handle list/dict returns from transit analysis functions
            if isinstance(fallback, list) and len(fallback) > 0 and isinstance(fallback[0], dict):
                # Transit analysis functions return [{"summary": {...}, "container_details": [...}}]
                item = fallback[0]
                if "summary" in item and "container_details" in item:
                    summary = item["summary"]
                    containers = item["container_details"]
                    
                    # Format summary message
                    if "ocean_bl_number" in summary:
                        msg = f"Transit Analysis for Ocean BL: {summary['ocean_bl_number']}\n\n"
                    elif "po_number" in summary:
                        msg = f"Transit Analysis for PO: {summary['po_number']}\n\n"
                    else:
                        msg = "Transit Analysis Summary:\n\n"
                    
                    msg += f"Total Containers: {summary.get('total_containers', 0)}\n"
                    msg += f"Arrived: {summary.get('arrived_containers', 0)}, In Transit: {summary.get('in_transit_containers', 0)}\n"
                    msg += f"Average Transit Time: {summary.get('avg_transit_days', 'N/A')} days\n"
                    msg += f"Average Delay: {summary.get('avg_delay_days', 'N/A')} days\n"
                    msg += f"Delayed Containers: {summary.get('delayed_containers', 0)}\n"
                    msg += f"On-Time/Early: {summary.get('on_time_or_early_containers', 0)}"
                    
                    return {
                        "response": sanitize_response(msg),
                        "observation": "Fallback response due to agent error",
                        "table": containers,  # Container details as table
                        "mode": "router-fallback"
                    }
            
            # Default string response
            fallback_str = str(fallback) if not isinstance(fallback, str) else fallback
            return {
                "response": sanitize_response(fallback_str),
                "observation": "Fallback response due to agent error",
                "table": [],
                "mode": "router-fallback"
            }
        except Exception as fallback_exc:
            logger.error(f"Router fallback also failed: {fallback_exc}")
            raise HTTPException(status_code=500, detail=f"Agent failed: {exc}")
