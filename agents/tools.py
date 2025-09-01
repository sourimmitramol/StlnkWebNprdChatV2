# agents/tools.py
import logging
import re
from datetime import datetime, timedelta
from langchain_community.agent_toolkits import create_sql_agent
import pandas as pd
from fuzzywuzzy import process
from langchain.agents import Tool
from services.vectorstore import get_vectorstore
from utils.container import extract_container_number
from utils.logger import logger
from services.azure_blob import get_shipment_df
from utils.misc import to_datetime, clean_container_number
from agents.prompts import map_synonym_to_column
from agents.prompts import COLUMN_SYNONYMS
from services.vectorstore import get_vectorstore
from langchain_community.chat_models import AzureChatOpenAI
from config import settings
import sqlite3
from sqlalchemy import create_engine
# ------------------------------------------------------------------
# Helper – give every tool a clean copy of the DataFrame
# ------------------------------------------------------------------
def _df() -> pd.DataFrame:
    return get_shipment_df()

def map_synonym_to_column(term: str) -> str:
    term = term.lower().replace("_", " ").strip()
    return COLUMN_SYNONYMS.get(term, term)
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
        rows = df[df["container_number"].str.contains(container_no, case=False, na=False)]

    if rows.empty:
        return f"No data found for container {container_no}."

    row = rows.iloc[0]

    # -------------------------------------------------
    # Build the milestone list (only keep non-null dates)
    # -------------------------------------------------
    row_milestone_map = [
        ("<strong>Departed From</strong> ", row.get("load_port"), row.get('atd_lp')),
        ("<strong>Final Load Port Arrival</strong>", row.get("final_load_port"), row.get('ata_flp')),
        ("<strong>Final Load Port Departure</strong>", row.get("final_load_port"), row.get('atd_flp')),
        ("<strong>Reached at Discharge Port</strong>", row.get("discharge_port"), row.get('ata_dp')),
        ("<strong>Reached at Last CY</strong>", row.get("last_cy_location"), row.get('equipment_arrived_at_last_cy')),
        ("<strong>Out Gate at Last CY</strong>", row.get("out_gate_at_last_cy_lcn"), row.get('out_gate_at_last_cy')"),
        ("<strong>Delivered at</strong>", row.get("delivery_date_to_consignee_lcn"), row.get('delivery_date_to_consignee')),
        ("<strong>Container Returned to</strong>", row.get("empty_container_return_lcn"), row.get('empty_container_return_date')),
    ]

    c_df = pd.DataFrame(row_milestone_map)
    # print(c_df)

    f_df = c_df.dropna(subset=2)
    # print(f_df)

    f_line = f_df.iloc[0]
    l_line = f_df.iloc[-1]
    # print(l_line)

    # print("Bot Answer:_____")
    res = f"The <con>{container_no}</con> {l_line.get(0)} {l_line.get(1)} on {l_line.get(2)}\n\n{f_df.to_string(index=False, header=False)}."
    # print(res)

    # return "\n".join(status_lines)
    return res


# ------------------------------------------------------------------
# 2️⃣ Delayed Containers (X days)
# ------------------------------------------------------------------
def ensure_datetime(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Convert specified columns to datetime if not already."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def get_delayed_containers(query: str) -> str:
    import re
    df = _df()
    df = ensure_datetime(df, ["eta_dp", "ata_dp"])
    if "delay_days" not in df.columns:
        df["delay_days"] = (df["ata_dp"] - df["eta_dp"]).dt.days

    m = re.search(r"delayed by (\d+) days", query, re.IGNORECASE)
    days = int(m.group(1)) if m else 7

    delayed_df = df[df["delay_days"] >= days]
    if delayed_df.empty:
        return "No container is delayed."
    cols = ["container_number", "discharge_port", "eta_dp", "ata_dp", "delay_days"]
    cols = [c for c in cols if c in delayed_df.columns]
    delayed_df = delayed_df[cols].sort_values("delay_days", ascending=False).head(15)
    delayed_df = ensure_datetime(delayed_df, ["eta_dp", "ata_dp"])
    delayed_df["eta_dp"] = delayed_df["eta_dp"].dt.strftime("%Y-%m-%d")
    delayed_df["ata_dp"] = delayed_df["ata_dp"].dt.strftime("%Y-%m-%d")
   # return f"Containers delayed by at least {days} days:\n\n{delayed_df.to_string(index=False)}"
    return delayed_df.to_dict(orient="records")


# ------------------------------------------------------------------
# 3️⃣ Upcoming Arrivals (next X days)
# ------------------------------------------------------------------
def get_upcoming_arrivals(query: str) -> str:
    """
    List containers scheduled to arrive within the next X days.
    Input: Query should specify 'next <number> days' or 'upcoming <number> days'.
    Output: Table of containers with ETA in the specified window.
    Defaults to 7 days if not specified.
    """
    m = re.search(r"(?:next|upcoming)\s+(\d+)\s+days", query, re.IGNORECASE)
    days = int(m.group(1)) if m else 7

    df = _df()
    eta_col = next((c for c in ["eta_dp", "eta", "estimated_time_arrival"] if c in df.columns), None)
    if not eta_col:
        return "ETA column not found in the data."

    df = ensure_datetime(df, [eta_col])
    today = pd.Timestamp.today().normalize()
    future = today + pd.Timedelta(days=days)

    upcoming = df[(df[eta_col] >= today) & (df[eta_col] <= future)]
    if upcoming.empty:
        return f"No containers scheduled to arrive in the next {days} days."

    cols = ["container_number", "discharge_port", eta_col]
    upcoming = upcoming[cols].sort_values(eta_col).head(15)
    upcoming = ensure_datetime(upcoming, [eta_col])
    upcoming[eta_col] = upcoming[eta_col].dt.strftime("%Y-%m-%d")
    return upcoming.to_string(index=False)


# ------------------------------------------------------------------
# 4️⃣ Container ETA/ATA (single container)
# ------------------------------------------------------------------
def get_container_eta(query: str) -> str:
    """
    Return ETA and ATA details for a specific container.
    Input: Query must mention a container number (partial or full).
    Output: ETA/ATA and port details for the container.
    If not found, prompts for a valid container number.
    """
    cont = extract_container_number(query)
    if not cont:
        return "Please mention a container number."

    df = _df()
    df = ensure_datetime(df, ["eta_dp", "ata_dp"])
    row = df[df["container_number"].astype(str).str.contains(cont, case=False, na=False)]
    if row.empty:
        return f"No data for container {cont}."

    row = row.iloc[0]
    cols = ["container_number", "discharge_port", "eta_dp", "ata_dp"]
    cols = [c for c in cols if c in row.index]
    out = row[cols].to_frame().T
    out = ensure_datetime(out, ["eta_dp", "ata_dp"])
    out["eta_dp"] = out["eta_dp"].dt.strftime("%Y-%m-%d")
    out["ata_dp"] = out["ata_dp"].dt.strftime("%Y-%m-%d")
    return out.to_string(index=False)


# ------------------------------------------------------------------
# 5️⃣ Arrivals By Port / Country
# ------------------------------------------------------------------
def get_arrivals_by_port(query: str) -> str:
    """
    Find containers arriving at a specific port or country.
    Input: Query should specify the port or country name (supports synonyms).
    Output: Table of containers arriving at the specified location.
    If not found, prompts for a valid port/country.
    """
    m = re.search(r"in\s+([A-Za-z\s]+)", query, re.IGNORECASE)
    if not m:
        return "Please tell me the port or country you are interested in."
    place = m.group(1).strip()

    # Use synonym mapping for port name
    port_column = map_synonym_to_column("lp")  # Always map "lp" to "load_port"
    df = _df()
    df = ensure_datetime(df, ["eta_dp"])
    if port_column in df.columns:
        filtered = df[df[port_column].astype(str).str.contains(place, case=False, na=False)]
    else:
        filtered = df[df.apply(lambda r: place.lower() in str(r).lower(), axis=1)]

    if filtered.empty:
        return f"No containers found arriving in {place}."

    cols = ["container_number", port_column, "eta_dp"]
    cols = [c for c in cols if c in filtered.columns]
    filtered = ensure_datetime(filtered, ["eta_dp"])
    filtered["eta_dp"] = filtered["eta_dp"].dt.strftime("%Y-%m-%d")
    return filtered[cols].head(10).to_string(index=False)


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
        "location": r"\b(carrier_vehicle_load_lcn|vehicle_departure_lcn|vehicle_arrival_lcn|carrier_vehicle_unload_lcn|out_gate_location|equipment_arrival_at_last_lcn|out_gate_at_last_cy_lcn|delivery_location_to_consignee|empty_container_return_lcn)\b",
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
        cols = [c for c in ["carrier_vehicle_load_lcn", "vehicle_departure_lcn",
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
    days = int(m.group(1)) if m else 7

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

    result = filtered[cols].drop_duplicates().head(15).to_string(index=False)
    return f"Upcoming POs (next {days} days):\n\n{result}"


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
    List containers arriving soon (ETA window, ATA is null).
    Input: Query should specify 'next <number> days'.
    Output: Table of containers with ETA in the specified window and not yet arrived.
    Defaults to 7 days if not specified.
    """
    m = re.search(r"next\s+(\d+)\s+days", query, re.IGNORECASE)
    days = int(m.group(1)) if m else 7

    df = _df()
    df = ensure_datetime(df, ["eta_dp", "ata_dp"])

    today = pd.Timestamp.today().normalize()
    future = today + pd.Timedelta(days=days)

    mask = (df["ata_dp"].isna()) & (df["eta_dp"] >= today) & (df["eta_dp"] <= future)
    subset = df[mask]

    if subset.empty:
        return f"No containers arriving in the next {days} days (or they have already arrived)."

    subset = ensure_datetime(subset, ["eta_dp"])
    subset["eta_dp"] = subset["eta_dp"].dt.strftime("%Y-%m-%d")
    return subset[["container_number", "eta_dp"]].head(10).to_string(index=False)
from agents.prompts import map_synonym_to_column, ROBUST_COLUMN_MAPPING_PROMPT

def answer_with_column_mapping(query: str) -> str:
    """
    Uses the robust column mapping prompt and synonym mapping to interpret user queries.
    Maps synonyms to canonical column names and answers using the correct column.
    """
    # Example: extract possible column names from the query using synonyms
    import re
    # Try to find all possible column mentions in the query
    possible_columns = []
    for synonym in map_synonym_to_column.__globals__['COLUMN_SYNONYMS']:
        if re.search(rf"\b{re.escape(synonym)}\b", query, re.IGNORECASE):
            possible_columns.append(synonym)
    if not possible_columns:
        return (
            "Thought: Could not identify a column or field in the query using the provided synonyms.\n"
            "Final Answer: Please specify which field or column you want information about, using any of the synonyms listed in the prompt."
        )
    # Map all found synonyms to canonical columns
    canonical_columns = [map_synonym_to_column(col) for col in possible_columns]
    df = _df()
    missing = [col for col in canonical_columns if col not in df.columns]
    if missing:
        return (
            f"Thought: The following columns were not found in the data after synonym mapping: {', '.join(missing)}.\n"
            f"Final Answer: Please check your query and use the correct field names or synonyms as listed in the prompt."
        )
    # For demonstration, show top 5 values for each column
    details = []
    for col in canonical_columns:
        vc = df[col].value_counts().head(5)
        if not vc.empty:
            details.append(f"Top values for '{col}':\n" + "\n".join([f"{val}: {cnt}" for val, cnt in vc.items()]))
        else:
            details.append(f"No data available for column '{col}'.")
    return (
        f"Thought: Mapped user query columns {possible_columns} to canonical columns {canonical_columns} using the provided prompt and synonyms.\n"
        f"Final Answer:\n" + "\n\n".join(details) +
        "\n\nThese results are based on robust synonym mapping as described in the prompt."
    )
def get_top_values_for_column(query: str) -> str:
    """
    Get the top 5 most frequent values for a specified column.
    Input: Query should specify the column name or synonym.
    Output: Top 5 values and their counts.
    """
    import re
    match = re.search(r"top values for ([\w\s#]+)", query, re.IGNORECASE)
    if not match:
        return "Please specify which column you want the top values for."
    user_column = match.group(1).strip()
    column = map_synonym_to_column(user_column)
    df = _df()
    if column not in df.columns:
        return f"Column '{user_column}' not found in the data (after mapping to '{column}')."
    vc = df[column].value_counts().head(5)
    if vc.empty:
        return f"No data available for column '{column}'."
    details = "\n".join([f"{val}: {cnt}" for val, cnt in vc.items()])
    return f"Top values for '{column}':\n{details}"

def get_load_port_for_container(query: str) -> str:
    """
    Get the load port details for a specific container.
    Input: Query must mention a container number (partial or full).
    Output: Load port for the container.
    """
    container_no = extract_container_number(query)
    if not container_no:
        return "Please specify a valid container number to get load port details."
    df = _df()
    port_col = map_synonym_to_column("lp")  # Always maps "lp" to "load_port"
    if port_col not in df.columns:
        return "Load port information is not available in the data."
    row = df[df["container_number"].astype(str).str.contains(container_no, case=False, na=False)]
    if row.empty:
        return f"No data found for container {container_no}."
    load_port = row.iloc[0][port_col]
    return f"The LP (Load Port) for container {container_no} is {load_port}. This is the port where the container was loaded onto the vessel."

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

from langchain_community.chat_models import AzureChatOpenAI
from config import settings

def get_sql_agent():
    # Initialize your LLM (Azure OpenAI)
    llm = AzureChatOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
    )
    engine = get_blob_sql_engine()
    return create_sql_agent(llm, db=engine, agent_type="openai-tools", verbose=False)

def sql_query_tool(query: str) -> str:
    agent = get_sql_agent()
    try:
        result = agent.run(query)
        return f"Thought: Used SQL agent to answer the query.\nFinal Answer: {result}"
    except Exception as exc:
        return f"Error: SQL agent failed to answer the query: {exc}"

def get_blob_sql_engine():
    """
    Loads the shipment CSV from Azure Blob and creates an in-memory SQLite engine for SQL queries.
    """
    # Load DataFrame from Azure Blob
    df = get_shipment_df()
    # Create in-memory SQLite database
    conn = sqlite3.connect(":memory:")
    df.to_sql("shipment", conn, index=False, if_exists="replace")
    # Create SQLAlchemy engine from the SQLite connection
    engine = create_engine("sqlite://", creator=lambda: conn)
    return engine
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
        name="SQL Query Shipment Data",
        func=sql_query_tool,
        description="Execute SQL queries against the shipment data stored in an in-memory SQLite database."
    ),
]









