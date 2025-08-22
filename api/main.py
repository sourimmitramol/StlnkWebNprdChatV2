# api/main.py
import logging
import re
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

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
    q = body.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty query")

    try:
        # Use the Azure agent for answers instead of the tool router
        answer = AGENT.invoke(q)
        return {"response": answer, "mode": "agent"}
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

# add to api/main.py


@app.post("/ask_with_consignee")
def ask_with_consignee(body: QueryWithConsigneeBody):
    """
    Robust Prompt: You are a shipping data expert.
    For every query, filter the dataset so only rows containing any of the provided consignee codes
    (even if multiple codes are present in the field) are considered.
    If any consignee code from the comma-separated list is present anywhere in the 'consignee_code_multiple' column
    (including as part of a comma-separated list or inside parentheses), answer the question using only those rows.
    If no rows match, reply with a clear message.
    """

    q = body.question.strip()
    codes = [c.strip() for c in body.consignee_code.split(",") if c.strip()]
    if not q or not codes:
        raise HTTPException(status_code=400, detail="Both question and consignee_code are required.")

    # Get the DataFrame (replace with your actual DataFrame source)
    from services.azure_blob import get_shipment_df
    df = get_shipment_df().copy()

    # Build a regex pattern to match any code as a whole word, allowing for comma-separated, parentheses, or spaces
    pattern = r"|".join([rf"\b{re.escape(code)}\b" for code in codes])
    mask = df['consignee_code_multiple'].astype(str).apply(
        lambda x: bool(re.search(pattern, x))
    )
    filtered_df = df[mask]
    if filtered_df.empty:
        return {"response": f"No data found for consignee code(s): {', '.join(codes)}."}

    # Now answer the user's question using only the filtered DataFrame
    # For example, you can use your router or agent, passing the filtered_df as context
    # Here, we'll use a simple example with analyze_data_with_pandas
    try:
        # If you have a router or agent that accepts a DataFrame, pass filtered_df to it
        # Example: answer = route_query(q, df=filtered_df)
        # For demonstration, we'll just return the first few rows
        result = filtered_df.head(10).to_dict(orient="records")
        return {"response": result}
    except Exception as exc:
        return {"response": f"Error processing query: {exc}"}