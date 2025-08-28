# api/main.py
import logging
import re
import ast
from fastapi import FastAPI, HTTPException, Query, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List


from .schemas import AskRequest, QueryWithConsigneeBody
from agents.azure_agent import initialize_azure_agent
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
    answer_with_column_mapping,
    vector_search_tool,
    get_blob_sql_engine,
   # sql_query_tool  # Make sure this is defined in tools.py
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


def check_container_authorization(df, container_numbers: List[str], consignee_codes: List[str]):
    """Check if the container belongs to one of the authorized consignees"""
    authorized_df = filter_by_consignee(df, consignee_codes)
    container_mask = authorized_df['container_number'].isin(container_numbers)
    return authorized_df[container_mask], not authorized_df[container_mask].empty


@app.post("/ask")
def ask(body: QueryWithConsigneeBody):
    q = body.question.strip()
    consignee_codes = [c.strip() for c in body.consignee_code.split(",") if c.strip()]
    
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
        return {"response": f"No data found for consignee code(s): {', '.join(consignee_codes)}.", "mode": "agent"}

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
            return {
                "response": "You are not authorized to access information about the requested container(s). "
                           "Please verify the container numbers or your consignee codes.",
                "mode": "agent"
            }
    
    try:
        # Modify the question to include consignee context
        # consignee_context = f"For consignee codes {', '.join(consignee_codes)}: {q}"
        consignee_context = f"user: {q}"
        
        # Use the Azure agent for answers
        answer = AGENT.invoke(consignee_context)
        output = answer["output"]
        
        # Extract table data if present
        table_data = []
        
        # First try to extract JSON array from the text using regex
        json_match = re.search(r"(\[{.*?}\])", output, re.DOTALL)
        if json_match:
            try:
                # Try to parse the JSON array
                json_text = json_match.group(1)
                table_data = ast.literal_eval(json_text)
                # Remove the JSON part from the response text
                output = output.replace(json_text, "").strip()
                # Clean up any remaining markdown table syntax or brackets
                output = re.sub(r'```(json|python)?\s*', '', output)
                output = re.sub(r'\s*```', '', output)
                output = re.sub(r'\[\s*\]', '', output).strip()
            except (SyntaxError, ValueError) as e:
                logger.warning(f"Failed to parse JSON in response: {e}")
                
        # If no JSON found or parsing failed, look for markdown tables
        if not table_data and "```" in output:
            try:
                # Try to extract markdown code blocks that might contain data
                code_blocks = re.findall(r'```(?:json|python)?\s*([\s\S]*?)```', output)
                if code_blocks:
                    for block in code_blocks:
                        # Try to parse the code block as JSON or Python literal
                        try:
                            parsed_data = ast.literal_eval(block.strip())
                            if isinstance(parsed_data, list) and len(parsed_data) > 0 and isinstance(parsed_data[0], dict):
                                table_data = parsed_data
                                # Remove the code block from the output
                                output = output.replace(f"```{block}```", "").strip()
                                break
                        except (SyntaxError, ValueError):
                            continue
            except Exception as e:
                logger.warning(f"Failed to extract table data from markdown: {e}")
        
        # If using SQL query tool or vector search produced results, they might be in specific format
        # This additional check handles cases where the LLM didn't properly format the response
        if not table_data:
            try:
                # Check for SQL or vector search results in the standard tools output
                for tool_name in ["SQL Query Shipment Data", "Vector Search"]:
                    tool_pattern = fr"{tool_name}:\s*([\s\S]*?)(?:\n\n|$)"
                    tool_match = re.search(tool_pattern, output)
                    if tool_match:
                        tool_output = tool_match.group(1).strip()
                        try:
                            # Try to parse tool output as JSON or Python literal
                            if tool_output.startswith("[") and tool_output.endswith("]"):
                                parsed_tool_data = ast.literal_eval(tool_output)
                                if isinstance(parsed_tool_data, list) and len(parsed_tool_data) > 0:
                                    table_data = parsed_tool_data
                                    # Clean up the output by removing the tool result
                                    output = output.replace(tool_match.group(0), "").strip()
                                    break
                        except (SyntaxError, ValueError):
                            continue
            except Exception as e:
                logger.warning(f"Failed to extract tool output: {e}")
        
        # If we found table data, ensure it's JSON serializable
        if table_data:
            # Replace NaN, inf, -inf with None
            import pandas as pd
            import numpy as np
            import json
            
            # Helper function to convert non-serializable values
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
            
            # Ensure we can actually serialize it
            try:
                json.dumps(table_data)
            except TypeError:
                logger.error("Failed to serialize table data to JSON")
                table_data = []
        
        # Clean up the response text
        message = re.sub(r'\n\s*\n', '\n\n', output).strip()
        
        # Return formatted response with both the message and table data
        return {
            
            "response": message,
            "table": table_data,
            "mode": "agent"
        }
    except Exception as exc:
        logger.error(f"Agent failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent failed: {exc}")

# ----------------------------------------------------------------------
# Light‑weight wrappers for each tool - all now require consignee header
# ----------------------------------------------------------------------
@app.get("/tool/container_milestones")
def container_milestones(
    q: str = Query(..., description="e.g. 'container ABCD1234567'"),
    x_consignee_code: str = Header(..., description="Comma-separated list of authorized consignee codes")
):
    consignee_codes = [c.strip() for c in x_consignee_code.split(",") if c.strip()]
    if not consignee_codes:
        raise HTTPException(status_code=400, detail="Consignee code is required")
        
    # Extract container number from query
    from utils.container import extract_container_number
    container_no = extract_container_number(q)
    
    if not container_no:
        return {"result": "Please specify a valid container number."}
    
    # Check authorization
    from services.azure_blob import get_shipment_df
    df = get_shipment_df().copy()
    
    # Check if container belongs to authorized consignee
    container_df, is_authorized = check_container_authorization(df, [container_no], consignee_codes)
    
    if not is_authorized:
        return {"result": f"You are not authorized to access information about container {container_no}."}
        
    return {"result": get_container_milestones(q)}

# Similar modifications would be applied to all other tool endpoints
# For brevity, I'm only showing one example

# ----------------------------------------------------------------------
# Keep the existing ask_with_consignee endpoint
# ----------------------------------------------------------------------
@app.post("/ask_with_consignee")
def ask_with_consignee(body: QueryWithConsigneeBody):
    """
    Shipping data expert that only answers questions about containers 
    that belong to the specified consignee codes.
    
    If the user asks about specific containers, verify they belong to the authorized consignee
    before providing any information.
    """
    q = body.question.strip()
    consignee_codes = [c.strip() for c in body.consignee_code.split(",") if c.strip()]
    
    if not q or not consignee_codes:
        raise HTTPException(status_code=400, detail="Both question and consignee_code are required.")

    # Get the DataFrame from blob storage
    from services.azure_blob import get_shipment_df
    df = get_shipment_df().copy()
    
    # First, filter by the consignee codes the user is authorized to see
    pattern = r"|".join([rf"\b{re.escape(code)}\b" for code in consignee_codes])
    mask = df['consignee_code_multiple'].astype(str).apply(
        lambda x: bool(re.search(pattern, x))
    )
    authorized_df = df[mask]
    
    if authorized_df.empty:
        return {"response": f"No data found for consignee code(s): {', '.join(consignee_codes)}."}
    
    # Check if the query is about specific containers
    container_pattern = r'(?:container(?:s)?\s+(?:number(?:s)?)?(?:\s+is|\s+are)?\s+)?([A-Z]{4}\d{7}(?:\s*,\s*[A-Z]{4}\d{7})*)'
    container_match = re.search(container_pattern, q, re.IGNORECASE)
    
    if container_match:
        # Extract container numbers from the query
        container_str = container_match.group(1)
        requested_containers = [c.strip() for c in re.split(r'\s*,\s*', container_str)]
        
        # Check if these containers belong to the authorized consignees
        container_auth_mask = authorized_df['container_number'].isin(requested_containers)
        
        if not container_auth_mask.any():
            return {
                "response": "You are not authorized to access information about the requested container(s). "
                           "Please verify the container numbers or your consignee codes."
            }
        
        # Filter to only show data about the requested AND authorized containers
        filtered_df = authorized_df[container_auth_mask]
    else:
        # If no specific containers mentioned, use all data for authorized consignees
        filtered_df = authorized_df
    
    try:
        # Use the agent for more sophisticated responses
        consignee_context = f"For consignee codes {', '.join(consignee_codes)}: {q}"
        answer = AGENT.invoke(consignee_context)
        
        return {
            "response": answer["output"],
            "mode": "agent"
        }
    except Exception as exc:
        logger.error(f"Error processing query: {exc}", exc_info=True)
        
        # Fallback to simpler response if agent fails
        try:
            # Handle non-JSON compliant values
            filtered_df = filtered_df.replace([float('nan'), float('inf'), float('-inf')], None)
            
            # Convert numeric columns to objects and replace NaN with None
            for col in filtered_df.select_dtypes(include=['float64', 'int64']).columns:
                filtered_df[col] = filtered_df[col].astype('object').where(filtered_df[col].notna(), None)
            
            # Extract relevant fields based on query patterns
            result_fields = ['container_number']
            
            po_pattern = r'(po\s+number|purchase\s+order|order\s+number)'
            eta_pattern = r'(eta|estimated\s+time\s+of\s+arrival|arrival\s+date)'
            vessel_pattern = r'(vessel|ship)'
            
            if re.search(po_pattern, q, re.IGNORECASE):
                result_fields.extend(['po_number_multiple', 'consignee_code_multiple'])
            elif re.search(eta_pattern, q, re.IGNORECASE):
                result_fields.extend(['eta_dp', 'discharge_port', 'consignee_code_multiple'])
            elif re.search(vessel_pattern, q, re.IGNORECASE):
                result_fields.extend(['first_vessel_name', 'first_voyage_code', 'consignee_code_multiple'])
            else:
                result_fields.extend(['po_number_multiple', 'eta_dp', 'first_vessel_name', 'discharge_port', 'consignee_code_multiple'])
            
            existing_fields = [field for field in result_fields if field in filtered_df.columns]
            
            # Create a formatted response string
            if container_match:
                container_no = requested_containers[0]
                
                # Format response based on what was asked
                if re.search(po_pattern, q, re.IGNORECASE) and 'po_number_multiple' in filtered_df.columns:
                    po_numbers = filtered_df.iloc[0]['po_number_multiple']
                    po_list = [po.strip() for po in str(po_numbers).split(',') if po.strip()]
                    return {
                        "response": f"The PO numbers for container {container_no} are {', '.join(po_list)}.",
                        "mode": "agent"
                    }
                elif re.search(eta_pattern, q, re.IGNORECASE) and 'eta_dp' in filtered_df.columns:
                    eta = filtered_df.iloc[0]['eta_dp']
                    if pd.notnull(eta) and isinstance(eta, pd.Timestamp):
                        eta_str = eta.strftime('%Y-%m-%d')
                    else:
                        eta_str = str(eta)
                    return {
                        "response": f"The ETA for container {container_no} is {eta_str}.",
                        "mode": "agent"
                    }
                elif re.search(vessel_pattern, q, re.IGNORECASE) and 'first_vessel_name' in filtered_df.columns:
                    vessel = filtered_df.iloc[0]['first_vessel_name']
                    return {
                        "response": f"The vessel for container {container_no} is {vessel}.",
                        "mode": "agent"
                    }
            
            # Generic fallback
            result = filtered_df[existing_fields].head(1).to_dict(orient="records")[0]
            return {"response": str(result), "mode": "agent"}
            
        except Exception as inner_exc:
            logger.error(f"Fallback processing failed: {inner_exc}", exc_info=True)

            return {"response": f"Error processing query: {str(exc)}", "mode": "agent"}

