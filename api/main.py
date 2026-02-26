# api/main.py

import ast
import logging
import re
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from agents.azure_agent import initialize_azure_agent
from agents.memory_manager import memory_manager
from agents.router import route_query
from agents.tools import get_cargo_ready_date  # Add this
from agents.tools import get_hot_containers  # Add this
from agents.tools import (_df, analyze_data_with_pandas,
                          answer_with_column_mapping, check_arrival_status,
                          get_arrivals_by_port, get_blob_sql_engine,
                          get_container_carrier, get_container_etd,
                          get_container_milestones,
                          get_containers_arriving_soon, get_delayed_containers,
                          get_delayed_pos, get_field_info,
                          get_load_port_for_container, get_upcoming_arrivals,
                          get_upcoming_pos, get_vessel_info,
                          get_weekly_status_changes, lookup_keyword,
                          sql_query_tool, vector_search_tool)
from utils.container import extract_po_number

from .schemas import AskRequest, QueryWithConsigneeBody

logger = logging.getLogger("shipping_chatbot")

app = FastAPI(
    title="Shipping Chatbot API",
    version="1.0.0",
    description="FastAPI wrapper around the Azure‑OpenAI agent you built.",
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
    mask = (
        df["consignee_code_multiple"]
        .astype(str)
        .apply(lambda x: bool(re.search(pattern, x)))
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
        r"\s+for\s+(?:user|consignee)(?:\s+code[s]?)?\s+[\d,\s]+",
        "",
        response_text,
        flags=re.IGNORECASE,
    )

    # Pattern 2: Remove "user XXXXX:" at the beginning (e.g., "user 0000866: What are...")
    response_text = re.sub(
        r"^\s*(?:user|consignee)(?:\s+code[s]?)?\s+[\d,\s]+\s*:\s*",
        "",
        response_text,
        flags=re.IGNORECASE,
    )

    # Pattern 3: Remove standalone "for user/consignee code(s) XXXXX"
    response_text = re.sub(
        r"\s+for\s+(?:the\s+)?(?:user|consignee)(?:\s+code[s]?)?\s+[\d,\s]+",
        "",
        response_text,
        flags=re.IGNORECASE,
    )

    # Pattern 4: Clean up any double spaces or trailing/leading whitespace
    response_text = re.sub(r"\s+", " ", response_text).strip()

    return response_text


def infer_concept_used(tool_name: str, tool_input: str) -> str:
    """
    Infer the concept/strategy used based on the tool and input.
    Returns a human-readable description of the reasoning approach.
    """
    tool_lower = tool_name.lower()
    input_lower = tool_input.lower()

    # Date-based queries
    if any(
        month in input_lower
        for month in [
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ]
    ):
        date_concept = (
            "Time-based filtering with ata_dp, eta_dp and month/year analysis"
        )
    else:
        date_concept = None

    # Tool-specific concepts
    if "delayed" in tool_lower:
        return date_concept or "Delay detection using ETA vs ATA comparison"
    elif "milestone" in tool_lower:
        return "Container lifecycle tracking with milestone events"
    elif "arrival" in tool_lower or "eta" in tool_lower:
        return date_concept or "Arrival prediction using ETA data"
    elif "vessel" in tool_lower:
        return "Vessel information lookup from carrier data"
    elif "port" in tool_lower:
        return "Port-based filtering and route analysis"
    elif "carrier" in tool_lower:
        return "Carrier identification from shipment records"
    elif "po" in tool_lower or "purchase order" in tool_lower:
        return "Purchase order tracking based on ETA DP and PO number extraction"
    elif "hot container" in tool_lower:
        return "Priority container identification based on urgency metrics"
    elif "cargo ready" in tool_lower:
        return "Cargo readiness tracking with timeline analysis"
    elif "vector" in tool_lower or "search" in tool_lower:
        return "Semantic search using AI embeddings"
    elif "sql" in tool_lower:
        return "Database query execution with SQL"
    else:
        return date_concept or "Data retrieval and filtering from shipment records"


def enrich_observation_with_concept(observation_list):
    """
    Add 'concept_used' field to each step in the observation.
    """
    enriched = []
    for step in observation_list:
        if isinstance(step, tuple) and len(step) == 2:
            action, result = step
            # Convert action to dict if it has attributes
            if hasattr(action, "__dict__"):
                action_dict = {
                    "tool": action.tool,
                    "tool_input": action.tool_input,
                    "log": action.log,
                    "type": "AgentAction",
                }
                # Add concept_used field
                action_dict["concept_used"] = infer_concept_used(
                    action.tool, action.tool_input
                )
                enriched.append([action_dict, result])
            else:
                enriched.append(step)
        else:
            enriched.append(step)
    return enriched


def check_container_authorization(
    df, container_numbers: List[str], consignee_codes: List[str]
):
    """Check if the container belongs to one of the authorized consignees"""
    authorized_df = filter_by_consignee(df, consignee_codes)
    container_mask = authorized_df["container_number"].isin(container_numbers)
    return authorized_df[container_mask], not authorized_df[container_mask].empty


# ...existing code...


@app.post("/ask")
def ask(body: QueryWithConsigneeBody):
    q = body.question.strip()
    session_id = body.session_id  # Get session_id from request
    # consignee_codes = [c.strip() for c in body.consignee_code.split(",") if c.strip()]
    consignee_codes = list(
        dict.fromkeys(c.strip() for c in body.consignee_code.split(",") if c.strip())
    )
    # print(consignee_codes)
    logger.info(f"User_Query: {q}, c_codes: {consignee_codes}, session: {session_id}")
    logger.info(
        f"[DEBUG] Received question type: {type(q)}, value: {q!r}, length: {len(q)}"
    )

    # Get or create memory for this session
    memory = memory_manager.get_memory(session_id)
    memory_manager.set_consignee_context(session_id, consignee_codes)

    # STEP 1: Extract entities from current question first (to catch new entities)
    memory_manager.extract_and_track_entities(session_id, q)

    # STEP 2: Resolve entity references (like "this container" -> "container MSBU4522691")
    # This will also extract entities from chat history if needed
    resolved_query = memory_manager.resolve_entity_references(session_id, q)
    if resolved_query != q:
        logger.info(f"✅ Entity resolved: '{q}' -> '{resolved_query}'")
        q = resolved_query  # Use resolved query for all subsequent processing
    else:
        logger.debug(f"No entity references found in: '{q}'")

    if not q:
        raise HTTPException(status_code=400, detail="Empty question")

    if not consignee_codes:
        raise HTTPException(status_code=400, detail="Consignee code is required")

    # Handle simple conversational queries before invoking agent to avoid parsing errors
    q_lower = q.lower().strip()
    logger.info(f"[DEBUG] q_lower: {q_lower!r}")

    # Greetings
    greetings = [
        "hi",
        "hello",
        "hey",
        "gm",
        "good morning",
        "good afternoon",
        "good evening",
        "hola",
    ]
    if any(
        q_lower == word
        or q_lower.startswith(word + " ")
        or q_lower.startswith(word + ",")
        for word in greetings
    ):
        logger.info(f"[DEBUG] Matched greeting pattern")
        response_text = (
            "Hello! I'm MCS AI, your shipping assistant. How can I help you today?"
        )
        # Save to memory for direct responses
        memory.save_context({"input": q}, {"output": response_text})
        return {
            "response": response_text,
            "observation": [],
            "table": [],
            "mode": "direct",
            "session_id": session_id,
        }

    # Thanks - ONLY match if it's a standalone thank you message, NOT part of shipping query
    thanks_words = ["thank", "thx", "thanks", "thank you", "ty", "much appreciated"]
    shipping_keywords = [
        "shipped",
        "container",
        "quantity",
        "cargo",
        "po",
        "eta",
        "vessel",
        "port",
        "delivery",
    ]

    # Only trigger thanks response if NO shipping keywords are present
    if any(word in q_lower for word in thanks_words) and not any(
        kw in q_lower for kw in shipping_keywords
    ):
        logger.info(f"[DEBUG] Matched thanks pattern (no shipping keywords)")
        response_text = "You're very welcome! Always happy to help. – MCS AI"
        memory.save_context({"input": q}, {"output": response_text})
        return {
            "response": response_text,
            "observation": [],
            "table": [],
            "mode": "direct",
            "session_id": session_id,
        }

    # How are you
    if "how are you" in q_lower or "how r u" in q_lower or "how are u" in q_lower:
        logger.info(f"[DEBUG] Matched 'how are you' pattern")
        response_text = "I'm doing great, thanks for asking! How can I assist you with your shipping needs today?"
        memory.save_context({"input": q}, {"output": response_text})
        return {
            "response": response_text,
            "observation": [],
            "table": [],
            "mode": "direct",
            "session_id": session_id,
        }

    # Who are you
    if (
        "who are you" in q_lower
        or "your name" in q_lower
        or "what is your name" in q_lower
        or "what's your name" in q_lower
    ):
        logger.info(f"[DEBUG] Matched 'who are you' pattern")
        response_text = "I'm MCS AI, your AI-powered shipping assistant. I can help you track containers, POs, and more."
        memory.save_context({"input": q}, {"output": response_text})
        return {
            "response": response_text,
            "observation": [],
            "table": [],
            "mode": "direct",
            "session_id": session_id,
        }

    # Farewells
    if any(
        word in q_lower
        for word in [
            "bye",
            "goodbye",
            "see you",
            "take care",
            "cya",
            "see ya",
            "good bye",
        ]
    ):
        response_text = "Goodbye! Have a wonderful day ahead. – MCS AI"
        memory.save_context({"input": q}, {"output": response_text})
        return {
            "response": response_text,
            "observation": [],
            "table": [],
            "mode": "direct",
            "session_id": session_id,
        }

    # Get the DataFrame from blob storage
    from services.azure_blob import get_shipment_df

    df = get_shipment_df().copy()

    # Filter by consignee codes
    authorized_df = filter_by_consignee(df, consignee_codes)

    if authorized_df.empty:
        response_text = "No data found."
        memory.save_context({"input": q}, {"output": response_text})
        return {
            "response": response_text,
            "table": [],
            "mode": "agent",
            "session_id": session_id,
        }

    # Extract container numbers from the query if present
    container_pattern = r"(?:container(?:s)?\s+(?:number(?:s)?)?(?:\s+is|\s+are)?\s+)?([A-Z]{4}\d{7}(?:\s*,\s*[A-Z]{4}\d{7})*)"
    container_match = re.search(container_pattern, q, re.IGNORECASE)

    if container_match:
        # Extract container numbers
        container_str = container_match.group(1)
        requested_containers = [c.strip() for c in re.split(r"\s*,\s*", container_str)]

        # Check if these containers belong to the authorized consignees
        _, is_authorized = check_container_authorization(
            df, requested_containers, consignee_codes
        )
        if not is_authorized:
            response_text = "No data found."
            memory.save_context({"input": q}, {"output": response_text})
            return {
                "response": response_text,
                "table": [],
                "mode": "agent",
                "session_id": session_id,
            }

    # ---------- NEW: helper to pull milestone Observation ----------
    # Include actual tool names and normalize to lowercase for matching.
    MILESTONE_TOOL_NAMES = {
        n.lower()
        for n in [
            "Get Container Milestones",
            "get_container_milestones",
            "Milestone Lookup",
            "Check Arrival Status",
            "check_arrival_status",
            "Get Container Status",
        ]
    }

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
                tool = getattr(action, "tool", None) or getattr(
                    action, "tool_name", None
                )
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

    def _summarize_pandas_row(row: dict, original_query: str) -> str:
        """Build a natural-language summary from a single pandas-fallback row.

        Keeps this logic lightweight and defensive so it can be reused for
        queries like service contract or job number lookups without affecting
        the table data.
        """
        if not isinstance(row, dict):
            return ""

        parts = []

        original_lower = (original_query or "").lower()

        container = row.get("container_number")
        service_contract = row.get("service_contract_number")
        booking = row.get("booking_number_multiple")
        po_numbers = row.get("po_number_multiple")
        bl_numbers = row.get("ocean_bl_no_multiple")
        load_port = row.get("load_port")
        discharge_port = row.get("discharge_port")
        job_no = row.get("job_no")
        # Prefer actual arrival date, then ETA, then revised ETA
        date_at_port = row.get("ata_dp") or row.get("eta_dp") or row.get("revised_eta")

        # If the user explicitly asked for job number, prioritise that in the
        # summary so questions like "What is its job number" are answered
        # directly.
        if any(phrase in original_lower for phrase in ["job number", "job no", "job_no"]):
            if container and job_no:
                parts.append(
                    f"The job number for container {container} is {job_no}."
                )
            elif job_no:
                parts.append(f"The job number is {job_no}.")

            # Optionally add a little context if available
            if container and booking:
                parts.append(f"It is associated with booking {booking}.")

            if discharge_port and date_at_port:
                parts.append(
                    f"It is at discharge port {discharge_port} on {date_at_port}."
                )
            elif discharge_port:
                parts.append(f"Discharge port: {discharge_port}.")
            elif load_port:
                parts.append(f"Load port: {load_port}.")

            return " ".join(parts).strip()

        # Default behaviour for non job-number queries (e.g. service contract)
        if container and service_contract:
            parts.append(
                f"The service contract number for container {container} is {service_contract}."
            )
        elif container and "service contract" in original_lower:
            parts.append(
                f"No explicit service contract number was found for container {container}."
            )

        if container and booking:
            parts.append(f"It is associated with booking {booking}.")

        if po_numbers:
            parts.append(f"PO number(s): {po_numbers}.")

        if bl_numbers:
            parts.append(f"Ocean BL: {bl_numbers}.")

        if discharge_port and date_at_port:
            parts.append(
                f"It is at discharge port {discharge_port} on {date_at_port}."
            )
        elif discharge_port:
            parts.append(f"Discharge port: {discharge_port}.")
        elif load_port:
            parts.append(f"Load port: {load_port}.")

        return " ".join(parts).strip()

    # ---------------------------------------------------------------

    try:
        # Set consignee codes in thread-local storage so tools can access them
        # DO NOT include consignee codes in agent prompt - it confuses routing with long lists
        import threading

        threading.current_thread().consignee_codes = consignee_codes

        # Get conversation history from memory and add as context
        chat_history = memory.load_memory_variables({}).get("chat_history", [])

        # Build query with context from memory if history exists
        if chat_history:
            # Format last 2 exchanges (4 messages) for context
            context_parts = []
            for msg in chat_history[-4:]:
                # Handle different message formats (BaseMessage, tuple, or dict)
                if hasattr(msg, "type") and hasattr(msg, "content"):
                    role = "User" if msg.type == "human" else "Assistant"
                    content = msg.content
                elif isinstance(msg, tuple) and len(msg) >= 2:
                    role = "User" if msg[0] == "human" else "Assistant"
                    content = msg[1]
                else:
                    continue
                context_parts.append(f"{role}: {content}")

            if context_parts:
                context_str = "\n".join(context_parts)
                # Add note about entity resolution to help agent understand
                query_with_context = (
                    f"[Previous conversation for context:\n{context_str}]\n\n"
                    f"[NOTE: Entity references like 'this container', 'this PO' have already been resolved to actual identifiers]\n\n"
                    f"Current question: {q}"
                )
            else:
                query_with_context = q
        else:
            query_with_context = q

        try:
            # Pass query with context to existing agent
            result = AGENT.invoke({"input": query_with_context})
        except TypeError:
            result = AGENT.invoke({"input": query_with_context})

        # Save the original question and response to memory
        tracked_response = False
        if isinstance(result, dict) and "output" in result:
            output_text = result.get("output", "")
            if isinstance(output_text, str):
                memory.save_context({"input": q}, {"output": output_text})
                # Extract and track entities from both question and response
                memory_manager.extract_and_track_entities(session_id, q)
                memory_manager.extract_and_track_entities(session_id, output_text)
                tracked_response = True

        # Clear the context after use
        if hasattr(threading.current_thread(), "consignee_codes"):
            delattr(threading.current_thread(), "consignee_codes")

        if isinstance(result, str):
            result = {"output": result, "intermediate_steps": []}

        # Handle both string and non-string outputs (for return_direct=True tools)
        raw_output = result.get("output") or ""

        # If output is a list/dict (from return_direct=True), convert to JSON string for processing
        # but also store the original for table_data
        direct_data = None
        if isinstance(raw_output, (list, dict)):
            direct_data = raw_output
            import json

            output = json.dumps(raw_output, indent=2, default=str)
        elif isinstance(raw_output, str):
            output = raw_output.strip()
        else:
            output = str(raw_output)

        print(result.get("intermediate_steps"))
        milestone_obs = _extract_milestone_observation(
            result.get("intermediate_steps", [])
        )

        # Fallback: try to extract inline "Observation: ... [Final Answer: ...]" from the LLM output
        inline_obs = None
        try:
            m = re.search(
                r"(Observation:\s*.+?)(?:\n\s*Final Answer:|\Z)",
                output,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if m:
                inline_obs = m.group(1).strip()
        except Exception:
            pass

        # ---- NEW: Fallback when agent stops due to iteration/time limit ----
        if re.search(
            r"Agent stopped due to iteration limit or time limit\.",
            output,
            re.IGNORECASE,
        ):
            fallback = route_query(q, consignee_codes)  # Pass consignee codes

            # Handle list/dict returns from transit analysis functions
            if (
                isinstance(fallback, list)
                and len(fallback) > 0
                and isinstance(fallback[0], dict)
            ):
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
                    msg += (
                        f"Average Delay: {summary.get('avg_delay_days', 'N/A')} days\n"
                    )
                    msg += (
                        f"Delayed Containers: {summary.get('delayed_containers', 0)}\n"
                    )
                    msg += f"On-Time/Early: {summary.get('on_time_or_early_containers', 0)}"

                    return {
                        "response": sanitize_response(msg),
                        "observation": sanitize_response(msg),
                        "table": containers,  # Container details as table
                        "mode": "router-fallback",
                        "session_id": session_id,
                    }

            # Default string response
            fallback_str = str(fallback) if not isinstance(fallback, str) else fallback
            return {
                "response": sanitize_response(fallback_str),
                "observation": sanitize_response(fallback_str),
                "table": [],
                "mode": "router-fallback",
                "session_id": session_id,
            }
        # --------------------------------------------------------------------

        # -------- existing table extraction logic (operates on `output` only) ---
        table_data = []

        # If we have direct_data from return_direct=True, use it
        if direct_data and isinstance(direct_data, list):
            table_data = direct_data
        else:
            # Otherwise, try to extract from output string
            json_match = re.search(r"(\[{.*?}\])", output, re.DOTALL)
            if json_match:
                try:
                    json_text = json_match.group(1)
                    table_data = ast.literal_eval(json_text)
                    output = output.replace(json_text, "").strip()
                    output = re.sub(r"```(json|python)?\s*", "", output)
                    output = re.sub(r"\s*```", "", output)
                    output = re.sub(r"\[\s*\]", "", output).strip()
                except (SyntaxError, ValueError) as e:
                    logger.warning(f"Failed to parse JSON in response: {e}")

        if not table_data and "```" in output:
            try:
                code_blocks = re.findall(r"```(?:json|python)?\s*([\s\S]*?)```", output)
                if code_blocks:
                    for block in code_blocks:
                        try:
                            parsed_data = ast.literal_eval(block.strip())
                            if (
                                isinstance(parsed_data, list)
                                and parsed_data
                                and isinstance(parsed_data[0], dict)
                            ):
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
                    tool_pattern = rf"{tool_name}:\s*([\s\S]*?)(?:\n\n|$)"
                    tool_match = re.search(tool_pattern, output)
                    if tool_match:
                        tool_output = tool_match.group(1).strip()
                        try:
                            if tool_output.startswith("[") and tool_output.endswith(
                                "]"
                            ):
                                parsed_tool_data = ast.literal_eval(tool_output)
                                if (
                                    isinstance(parsed_tool_data, list)
                                    and parsed_tool_data
                                ):
                                    table_data = parsed_tool_data
                                    output = output.replace(
                                        tool_match.group(0), ""
                                    ).strip()
                                    break
                        except (SyntaxError, ValueError):
                            continue
            except Exception as e:
                logger.warning(f"Failed to extract tool output: {e}")

        if table_data:
            import json

            import numpy as np
            import pandas as pd

            def clean_non_serializable(obj):
                if isinstance(obj, dict):
                    return {k: clean_non_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_non_serializable(i) for i in obj]
                elif pd.isna(obj) or obj in [float("inf"), float("-inf")]:
                    return None
                elif isinstance(obj, pd.Timestamp):
                    return obj.strftime("%Y-%m-%d")
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
        is_milestone_response = False
        if milestone_obs:
            observation_text = str(milestone_obs).strip()

            # If the LLM produced a specially formatted milestone answer
            # ("Milestones (verbatim)" + "Desired result"), surface that
            # full text to the client. Otherwise, fall back to the raw
            # milestone observation string.
            if isinstance(raw_output, str) and "Milestones (verbatim)" in raw_output:
                message = raw_output.strip()
            else:
                message = observation_text

            is_milestone_response = True
        elif inline_obs:
            observation_text = inline_obs
            message = inline_obs
        else:
            message = (
                re.sub(r"\n\s*\n", "\n\n", output).strip()
                or "No results found for the given query."
            )

        observation = result.get("intermediate_steps", [])

        # Enrich observation with concept_used field
        enriched_observation = enrich_observation_with_concept(observation)

        # Bypass sanitize_response for milestone queries to preserve newline formatting
        response_message = (
            message if is_milestone_response else sanitize_response(message)
        )

        # ============ TRACK ENTITIES FROM RESPONSE (FALLBACK) ============
        # CRITICAL: Extract entities from agent response to maintain conversation context
        # This ensures "those containers" references work in follow-up questions
        # Only run if not already tracked above (to avoid duplicate processing)
        if not tracked_response and message:
            memory_manager.extract_and_track_entities(session_id, message)
            memory.save_context({"input": q}, {"output": message})
        # ==================================================================

        # ============ PANDAS CODE GENERATION FALLBACK ============
        # Detect if agent couldn't answer properly and fall back to pandas code generation
        should_use_pandas_fallback = False
        
        # High-priority rule: job number queries should be answered via
        # pandas analysis since there is no dedicated job_no tool.
        try:
            q_lower_for_fallback = q.lower()
        except Exception:
            q_lower_for_fallback = str(q).lower()

        if re.search(r"\bjob(?:_no| number| no)?\b", q_lower_for_fallback):
            should_use_pandas_fallback = True
            logger.info("[PANDAS FALLBACK] Triggered: job number query -> route to pandas engine")
        
        # Condition 1: Response contains phrases indicating agent couldn't understand
        unable_to_answer_patterns = [
            r"i couldn't understand",
            r"i don't have enough information",
            r"please try rephrasing",
            r"i'm not sure",
            r"unable to answer",
            r"cannot determine",
            r"no information available",
            r"i don't know",
            r"not available",           # Catches "is not available", "are not available"
            r"not found",               # Catches "was not found", "were not found"
            r"no.*\bdata\b.*found",     # Catches "no data found"
            r"no.*\bresult",            # Catches "no results", "no result found"
            r"not explicitly listed",    # Catches "not explicitly listed in the provided information"
        ]
        
        message_lower = message.lower()
        if any(re.search(pattern, message_lower) for pattern in unable_to_answer_patterns):
            should_use_pandas_fallback = True
            logger.info("[PANDAS FALLBACK]Triggered: Agent couldn't understand query")
        
        # Condition 2: Router returned generic error message
        if "I couldn't understand your query" in message:
            should_use_pandas_fallback = True
            logger.info("[PANDAS FALLBACK] Triggered: Router returned error message")
        
        # Execute pandas fallback if conditions are met
        if should_use_pandas_fallback:
            from agents.azure_agent import pandas_code_generator_fallback

            logger.info(f"[PANDAS FALLBACK] Attempting pandas code generation for: {q}")
            pandas_result = pandas_code_generator_fallback(
                q, consignee_codes, return_table=True
            )

            # Normalize result into text + optional table
            pandas_text = None
            pandas_table = []
            if isinstance(pandas_result, dict):
                pandas_text = pandas_result.get("text", "")
                tbl = pandas_result.get("table")
                if isinstance(tbl, list):
                    pandas_table = tbl
            else:
                pandas_text = str(pandas_result)

            # For job-number style questions, if the query mentions a specific
            # container and the pandas result contains multiple containers,
            # narrow the table to rows for that container so the answer and
            # table stay aligned with the user's context.
            q_lower_for_pandas = q.lower()
            if (
                pandas_table
                and any(phrase in q_lower_for_pandas for phrase in ["job number", "job no", "job_no"])
            ):
                import re as _re

                # Try to find container ID in the resolved query first
                container_match = _re.search(r"\b[A-Z]{4}\d{7}\b", q, _re.IGNORECASE)
                target_container = None
                
                if container_match:
                    target_container = container_match.group(0).upper()
                    logger.info(f"[JOB NUMBER FILTER] Found container in query: {target_container}")
                else:
                    # Fallback: get the most recent container from memory
                    if session_id in memory_manager.memories:
                        recent_containers = memory_manager.memories[session_id]["entities"]["containers"]
                        if recent_containers:
                            target_container = recent_containers[-1]
                            logger.info(f"[JOB NUMBER FILTER] Using recent container from memory: {target_container}")
                
                # Filter table to only rows matching the target container
                if target_container:
                    filtered_rows = [
                        r for r in pandas_table 
                        if str(r.get("container_number", "")).upper() == target_container.upper()
                    ]
                    if filtered_rows:
                        logger.info(f"[JOB NUMBER FILTER] Filtered {len(pandas_table)} rows to {len(filtered_rows)} for container {target_container}")
                        pandas_table = filtered_rows
                    else:
                        logger.warning(f"[JOB NUMBER FILTER] No rows found for container {target_container}, keeping all {len(pandas_table)} rows")

            # If we have structured rows, build a concise natural-language
            # summary for the first row and prefer that over the generic
            # "Found N record(s)" text, while keeping the table unchanged.
            if pandas_table:
                nl_summary = _summarize_pandas_row(pandas_table[0], q)
                if nl_summary:
                    pandas_text = nl_summary

            # Check if pandas fallback succeeded (didn't return error message)
            if pandas_text and not pandas_text.startswith("Error") and not pandas_text.startswith("Unable"):
                logger.info("[PANDAS FALLBACK] Successfully generated response using pandas")
                return {
                    "response": sanitize_response(pandas_text),
                    "observation": "Pandas code generation fallback",
                    "table": pandas_table,
                    "mode": "pandas-fallback",
                    "session_id": session_id,
                }
            else:
                logger.warning(f"[PANDAS FALLBACK] Failed: {pandas_result}")
                # Continue with original response
        # =========================================================

        return {
            "response": response_message,  # now surfaces the detailed observation in Postman
            "observation": enriched_observation,  # explicit field for clients that need it - now includes concept_used
            "table": table_data,
            "mode": "agent",
            "session_id": session_id,
        }

    except Exception as exc:
        logger.error(f"Agent failed: {exc}", exc_info=True)

        # Try router fallback before failing
        try:
            fallback = route_query(q, consignee_codes)  # Pass consignee codes

            # Handle list/dict returns from transit analysis functions
            if (
                isinstance(fallback, list)
                and len(fallback) > 0
                and isinstance(fallback[0], dict)
            ):
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
                    msg += (
                        f"Average Delay: {summary.get('avg_delay_days', 'N/A')} days\n"
                    )
                    msg += (
                        f"Delayed Containers: {summary.get('delayed_containers', 0)}\n"
                    )
                    msg += f"On-Time/Early: {summary.get('on_time_or_early_containers', 0)}"

                    return {
                        "response": sanitize_response(msg),
                        "observation": "Fallback response due to agent error",
                        "table": containers,  # Container details as table
                        "mode": "router-fallback",
                        "session_id": session_id,
                    }

            # Default string response
            fallback_str = str(fallback) if not isinstance(fallback, str) else fallback
            return {
                "response": sanitize_response(fallback_str),
                "observation": "Fallback response due to agent error",
                "table": [],
                "mode": "router-fallback",
                "session_id": session_id,
            }
        except Exception as fallback_exc:
            logger.error(f"Router fallback also failed: {fallback_exc}")
            raise HTTPException(status_code=500, detail=f"Agent failed: {exc}")




# ----------------------------------------------------------------------
# Session Management Endpoints
# ----------------------------------------------------------------------
@app.post("/session/clear")
def clear_session(session_id: str):
    """Clear conversation memory for a specific session"""
    success = memory_manager.clear_session(session_id)
    if success:
        return {
            "message": "Session cleared successfully",
            "session_id": session_id,
        }
    else:
        return {
            "message": "Session not found or already cleared",
            "session_id": session_id,
        }


@app.get("/session/info/{session_id}")
def get_session_info(session_id: str):
    """Get information about a specific session"""
    info = memory_manager.get_session_info(session_id)
    if info:
        return info
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/sessions/count")
def get_active_sessions():
    """Get count of active sessions"""
    count = memory_manager.get_active_sessions_count()
    return {
        "active_sessions": count,
        "message": f"There are {count} active session(s)",
    }
