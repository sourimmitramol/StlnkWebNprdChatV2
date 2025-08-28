# api/main.py
import logging
import re
import ast
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from .schemas import AskRequest, QueryWithConsigneeBody  #AskRequest,
from agents.azure_agent import initialize_azure_agent
from agents.tools import (
    get_container_milestones,
    get_delayed_containers,
    get_upcoming_arrivals,
    get_container_eta,
    get_arrivals_by_port,
    lookup_keyword,
    analyze_data_with_pandas,  # <-- update this line
    get_field_info,
    get_vessel_info,
    get_upcoming_pos,
    get_delayed_pos,
    get_containers_arriving_soon,
)
from agents.router import route_query  # Import your router for detailed answers

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
# Generic Q&A endpoint – tries the agent first, then PandasAI
# ----------------------------------------------------------------------
@app.post("/ask")
def ask(body: AskRequest):
    q = body.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty query")

    try:
        # Use the Azure agent for answers instead of the tool router
        answer = AGENT.invoke(q)
        output = answer["output"]
        match = re.search(r"(\[.*\])", output)
        message = output
        json_array = []

        if match:
           # Extract JSON array
           json_array = ast.literal_eval(match.group(1))
           # Remove JSON part from original string
           message = output.replace(match.group(1), "").strip()
        
        return {"response": message,"table":json_array, "mode": "agent"}
    except Exception as exc:
        logger.error(f"Agent failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Agent failed: {exc}")


# ----------------------------------------------------------------------
# Light‑weight wrappers for each tool (useful for a UI that wants
# direct access without the LLM reasoning layer)
# ----------------------------------------------------------------------
@app.get("/tool/container_milestones")
def container_milestones(q: str = Query(..., description="e.g. 'container ABCD1234567'")):
    return {"result": get_container_milestones(q)}


@app.get("/tool/delayed_containers")
def delayed_containers(days: int = Query(7, ge=0)):
    return {"result": get_delayed_containers(f"delayed by {days}")}


@app.get("/tool/upcoming_arrivals")
def upcoming_arrivals(days: int = Query(7, ge=0)):
    return {"result": get_upcoming_arrivals(f"next {days} days")}


@app.get("/tool/container_eta")
def container_eta(q: str = Query(..., description="e.g. 'container ABCD1234567'")):
    return {"result": get_container_eta(q)}


@app.get("/tool/arrivals_by_port")
def arrivals_by_port(place: str = Query(..., description="City, port or country")):
    return {"result": get_arrivals_by_port(f"in {place}")}


@app.get("/tool/keyword_lookup")
def keyword_lookup(q: str = Query(...)):
    return {"result": lookup_keyword(q)}


@app.get("/tool/field_info")
def field_info(q: str = Query(...)):
    return {"result": get_field_info(q)}


@app.get("/tool/vessel_info")
def vessel_info(q: str = Query(...)):
    return {"result": get_vessel_info(q)}


@app.get("/tool/upcoming_pos")
def upcoming_pos(days: int = Query(7, ge=0)):
    return {"result": get_upcoming_pos(f"next {days} days")}


@app.get("/tool/delayed_pos")
def delayed_pos():
    return {"result": get_delayed_pos("")}


@app.get("/tool/containers_arriving_soon")
def containers_arriving_soon(days: int = Query(7, ge=0)):
    return {"result": get_containers_arriving_soon(f"next {days} days")}

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
        return {"response": "No data found.", "table": []}
   
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
                "response": "Not a valid shippement.", "table": []
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
        
        output = answer["output"]
        match = re.search(r"(\[.*\])", output)
        message = output
        json_array = []
        
        if match:
            # Extract JSON array
            json_array = ast.literal_eval(match.group(1))
            # Remove JSON part from original string
            message = output.replace(match.group(1), "").strip()
       
        return {
            "response": message,
            "table": json_array,
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
                        "table": [],
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
                        "table": [],
                        "mode": "agent"
                    }
                elif re.search(vessel_pattern, q, re.IGNORECASE) and 'first_vessel_name' in filtered_df.columns:
                    vessel = filtered_df.iloc[0]['first_vessel_name']
                    return {
                        "response": f"The vessel for container {container_no} is {vessel}.",
                        "table": [],
                        "mode": "agent"
                    }
           
            # Generic fallback
            result = filtered_df[existing_fields].head(1).to_dict(orient="records")[0]    
            output = str(result)
            match = re.search(r"(\[.*\])", output)
            message = output
            json_array = []    

            if match:
                # Extract JSON array
                json_array = ast.literal_eval(match.group(1))
                # Remove JSON part from original string
                message = output.replace(match.group(1), "").strip()            

            return {"response": message, "table": json_array, "mode": "agent"}
            
        except Exception as fallback_exc:
            logger.error(f"Fallback also failed: {fallback_exc}", exc_info=True)
            return {"response": f"Error: {fallback_exc}", "table": [], "mode": "error"}





