# agents/azure_agent.py
import logging
import threading
from typing import List, Tuple

from langchain.agents import AgentExecutor, Tool, create_structured_chat_agent
from langchain.agents.structured_chat.prompt import FORMAT_INSTRUCTIONS, PREFIX
from langchain.prompts import ChatPromptTemplate
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from agents.prompts import ROBUST_COLUMN_MAPPING_PROMPT
from agents.tools import get_blob_sql_engine
from config import settings

from .tools import TOOLS

logger = logging.getLogger("shipping_chatbot")


def initialize_azure_agent(
    tools: List[Tool] | None = None,
) -> Tuple[object, AzureChatOpenAI]:
    """Build the chat agent with Azure OpenAI.

    Notes:
    - Uses the non-deprecated `langchain_openai.AzureChatOpenAI`.
    - Uses the non-deprecated `create_structured_chat_agent` + `AgentExecutor`.
    - Preserves the previous return type/shape and `return_intermediate_steps=True` behavior.
    """
    if tools is None:
        tools = TOOLS

    llm = AzureChatOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
        temperature=0.03,  # Adjust temperature for creativity (0.0 = deterministic, 1.0 = creative)
    )
    # Build a structured-chat agent prompt (offline, no LangChain Hub dependency)
    # Required variables for create_structured_chat_agent: tools, tool_names, input, agent_scratchpad
    system_instructions = (
        f"{ROBUST_COLUMN_MAPPING_PROMPT}\n\n"
        "CRITICAL TOOL CALL RULES:\n"
        "- Make ONLY ONE tool call per user question\n"
        "- If user specifies a year (e.g., 'Oct 2025'), include that EXACT year in your FIRST and ONLY tool call\n"
        "- NEVER make a trial call without the year followed by another call with the year\n"
        "- NEVER modify dates provided by the user - pass them exactly as given\n"
        "- Example: User asks 'delayed in Oct 2025' → Make ONE call with 'containers delayed in Oct 2025'\n\n"
        f"{PREFIX}\n\n"
        "{tools}\n\n"
        f"{FORMAT_INSTRUCTIONS}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_instructions),
            # Structured chat agent expects `agent_scratchpad` as a STRING (not messages).
            ("human", "{input}\n\n{agent_scratchpad}"),
        ]
    )

    runnable_agent = create_structured_chat_agent(
        llm=llm, tools=tools or TOOLS, prompt=prompt
    )
    agent = AgentExecutor(
        agent=runnable_agent,
        tools=tools or TOOLS,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
    logger.info("AzureChatOpenAI ready")
    logger.info("Agent created with %d tools", len(tools))
    return agent, llm


def get_azure_embeddings() -> AzureOpenAIEmbeddings:
    """Return the Azure embeddings model – used by vectorstore and RAG."""
    return AzureOpenAIEmbeddings(
        azure_deployment=settings.AZURE_OPENAI_EMBEDDING_MODEL,
        openai_api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
    )


def initialize_sql_agent():
    """
    Initializes a SQL agent using the in-memory SQLite engine created from Azure Blob CSV.
    """
    # Set up your LLM (Azure OpenAI)
    llm = AzureChatOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
    )
    # Get the SQL engine from blob data
    engine = get_blob_sql_engine()
    # Create the SQL agent executor
    sql_agent_executor = create_sql_agent(
        llm, db=engine, agent_type="openai-tools", verbose=True
    )
    return sql_agent_executor


def pandas_code_generator_fallback(
    query: str,
    consignee_codes: list | None = None,
    retry_count: int = 0,
    return_table: bool = False,
) -> str | dict:
    """
    Pandas code generation fallback for queries not handled by predefined tools.
    
    This function generates and executes pandas code to answer queries when:
    - No predefined tool matches the user's query
    - The agent cannot find a suitable function
    - The query requires custom data analysis
    
    Features:
    - Automatic retry with query rephrasing on execution errors
    - Code syntax validation before execution
    - Safe execution environment with restricted builtins
    
    Args:
        query: User's natural language query
        consignee_codes: Optional list of consignee codes for access control
        retry_count: Internal counter for retry attempts (max: 1)
        
    Returns:
        If return_table is False (default): formatted response string.
        If return_table is True and the generated code produces a DataFrame 'result':
            {"text": summary_string, "table": list_of_row_dicts}
    """
    MAX_RETRIES = 1  # Allow one retry with rephrased query
    import pandas as pd
    import re
    from datetime import datetime, timedelta
    from agents.tools import _df
    from agents.prompts import (
        COLUMN_SYNONYMS, 
        INTENT_SYNONYMS, 
        parse_time_period,
        ROBUST_COLUMN_MAPPING_PROMPT
    )
    
    try:
        logger.info(f"[PANDAS FALLBACK] Generating code for query: {query}")
        
        # Get the DataFrame with consignee filtering
        import threading
        if consignee_codes:
            threading.current_thread().consignee_codes = consignee_codes
        
        df = _df()
        
        if df.empty:
            return "No data available to process your query."
        
        # Create comprehensive column descriptions using prompts.py knowledge
        column_descriptions = {
            "job_no": "Internal job number for the shipment (unique job identifier)",
            "container_number": "Unique container identifier (e.g., MEDU7986870)",
            "po_number_multiple": "Purchase order numbers (can be multiple, comma-separated)",
            "ocean_bl_no_multiple": "Ocean bill of lading numbers (can be multiple)",
            "booking_number_multiple": "Booking numbers for shipments",
            "consignee_code_multiple": "Customer/consignee codes with names",
            
            # Date columns - CRITICAL for queries
            "etd_lp": "Estimated Time of Departure from Load Port (departure date)",
            "atd_lp": "Actual Time of Departure from Load Port (actual departure)",
            "eta_dp": "Estimated Time of Arrival at Discharge Port (arrival date)",
            "ata_dp": "Actual Time of Arrival at Discharge Port (actual arrival) - NULL means in transit",
            "revised_eta": "Revised/Updated ETA (use this over eta_dp when available)",
            "eta_fd": "ETA at Final Destination",
            "revised_eta_fd": "Revised ETA at Final Destination",
            "predictive_eta_fd": "Predicted ETA at Final Destination",
            "delivery_date_to_consignee": "Actual delivery date to customer",
            "empty_container_return_date": "Date when empty container was returned",
            
            # Port and location columns
            "load_port": "Origin/Load Port (where container departed from)",
            "final_load_port": "Final Load Port (for transshipment)",
            "discharge_port": "Destination/Discharge Port (where container arrives)",
            "final_destination": "Final delivery destination (DC/warehouse)",
            "place_of_delivery": "Place where goods are delivered",
            "last_cy_location": "Last container yard location",
            
            # Carrier and vessel columns
            "final_carrier_name": "Shipping carrier/line name (e.g., MAERSK, CMA CGM)",
            "final_carrier_code": "Carrier code",
            "first_vessel_name": "First/Feeder vessel name",
            "first_vessel_code": "First vessel code",
            "final_vessel_name": "Final/Mother vessel name",
            "final_vessel_code": "Final vessel code",
            "first_voyage_code": "First voyage number",
            "final_voyage_code": "Final voyage number",
            
            # Status and flag columns
            "hot_container_flag": "Priority/urgent container flag (Y/Yes/True = hot)",
            "current_arrival_status": "Current arrival status",
            "current_departure_status": "Current departure status",
            "late_arrival_status": "Late arrival indicator",
            "late_booking_status": "Late booking indicator",
            
            # Supplier and customer columns
            "supplier_vendor_name": "Supplier/vendor name",
            "manufacturer_name": "Manufacturer name",
            "ship_to_party_name": "Ship-to party/recipient",
            
            # Cargo details
            "cargo_count": "Cargo count/quantity",
            "cargo_um": "Cargo unit of measure (CTN, PCS, etc.)",
            "cargo_detail_count": "Detailed cargo count",
            "detail_cargo_um": "Detailed cargo unit measure",
            "cargo_weight": "Total cargo weight",
            
            # Container details
            "container_type": "Container type (e.g., 40HC, 20GP)",
            "destination_service": "Destination service type",
            "transport_mode": "Transport mode (Ocean, Air, Rail)",
            "seal_number": "Container seal number (security seal identifier)",
            "seal_no": "Container seal number (alternative field name)",
            
            # Dates for gate and equipment movement
            "equipment_arrived_at_last_cy": "Equipment arrival at last container yard",
            "out_gate_at_last_cy": "Out gate at last container yard",
            "out_gate_date_from_dp": "Out gate date from discharge port",
            "carrier_vehicle_load_date": "Vehicle load date",
            "vehicle_departure_date": "Vehicle departure date",
            "vehicle_arrival_date": "Vehicle arrival date",
            
            # Free time and charges
            "detention_free_days": "Detention free days allowed",
            "demurrage_free_days": "Demurrage free days allowed",
        }
        
        # Build schema text with rich descriptions
        columns_info = []
        for col in df.columns[:60]:  # Increased to 60 for more context
            dtype = str(df[col].dtype)
            # Get description from our mapping or use generic
            desc = column_descriptions.get(col, "")
            
            # Add sample values only for key columns to save tokens
            if col in column_descriptions:
                sample_values = df[col].dropna().head(2).tolist()
                if desc:
                    columns_info.append(f"- {col} ({dtype}): {desc}\n  Sample: {sample_values}")
                else:
                    columns_info.append(f"- {col} ({dtype}): {sample_values}")
            else:
                # For other columns, just list them
                columns_info.append(f"- {col} ({dtype})")
        
        schema_text = "\n".join(columns_info)
        
        # Create reverse mapping for common user terms
        user_term_mapping = []
        seen_columns = set()
        for term, col in COLUMN_SYNONYMS.items():
            if col not in seen_columns and col in df.columns:
                user_term_mapping.append(f"  '{term}' → {col}")
                seen_columns.add(col)
                if len(seen_columns) >= 30:  # Limit to top 30
                    break
        
        term_mapping_text = "\n".join(user_term_mapping[:30])
        
        # Create LLM instance
        llm = AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
            temperature=0.0,
        )
        
        # Prompt for code generation with domain knowledge
        code_generation_prompt = f"""You are a pandas code generation expert for a shipping/logistics system.

USER QUERY: {query}

AVAILABLE DATAFRAME:
The DataFrame 'df' has {len(df)} rows and {len(df.columns)} columns with shipping/logistics data.

KEY COLUMNS WITH DESCRIPTIONS:
{schema_text}

USER TERM MAPPINGS (common terms users might use):
{term_mapping_text}

CRITICAL BUSINESS LOGIC FROM SHIPPING DOMAIN:
1. **Delay Calculation** (HIGHEST PRIORITY):
   - delay_days = (ata_dp - eta_dp).days where ata_dp IS NOT NULL
   - Positive delay = late arrival, Negative = early arrival
   - NULL ata_dp = still in transit (not arrived yet)
   - ONLY calculate delay for arrived containers (ata_dp NOT NULL)

2. **Container Status Logic**:
   - In transit: ata_dp IS NULL (not yet arrived at discharge port)
   - Arrived at DP: ata_dp IS NOT NULL (reached discharge port)
   - Delivered: delivery_date_to_consignee IS NOT NULL (delivered to customer)
   - Returned: empty_container_return_date IS NOT NULL (empty container returned)
   - At DP but NOT delivered: ata_dp NOT NULL AND delivery_date_to_consignee IS NULL

3. **Priority/Hot Containers**:
   - Filter: df['hot_container_flag'].isin(['Y', 'Yes', 'True', '1', 1, True])
   - Case-insensitive matching recommended

4. **Date Handling** (CRITICAL):
   - Always convert dates: pd.to_datetime(df['column'], errors='coerce')
   - Use datetime.now() for current date/time
   - For "today": datetime.now().date()
   - For date comparisons: ensure both sides are datetime
   - Respect verb tense in queries:
     * Future tense ("arriving", "going to") → use current/future dates
     * Past tense ("arrived", "reached") → use past dates

5. **Port and Location Matching**:
   - Ports are in format: "CITY, STATE(CODE)" e.g., "LOS ANGELES, CA(USLAX)"
   - Port codes: USNYC, USLAX, NLRTM, DEHAM, CNSHA, SGSIN, etc.
   - Use case-insensitive contains: df['discharge_port'].str.contains('PORT', case=False, na=False)
   - Search in: discharge_port (arrival), load_port (departure), final_destination

6. **Delay Intent Qualifiers**:
   - "more than N", "over N", "> N" → delay_days > N
   - "at least N", ">= N", "N or more", "minimum N" → delay_days >= N
   - "up to N", "no more than N", "within N", "maximum N" → delay_days <= N and > 0
   - "delayed by N", "N days late", plain "N days" → delay_days == N
   - If no number, return any positive delay (delay_days > 0)

7. **Aggregations and Grouping**:
   - Count containers: df['container_number'].nunique()
   - Count records: len(df) or df.shape[0]
   - Group by port: df.groupby('discharge_port')
   - Group by carrier: df.groupby('final_carrier_name')
   - Average delay: df['delay_days'].mean()
   - Median: df['column'].median()

8. **Multi-Value Columns** (IMPORTANT):
   - po_number_multiple: Can contain multiple POs comma-separated
   - ocean_bl_no_multiple: Multiple BLs comma-separated
   - consignee_code_multiple: Multiple consignees
   - Use .str.contains() for searching: df['po_number_multiple'].str.contains('PO123', na=False)

9. **Revised ETA Priority**:
   - Prefer revised_eta over eta_dp when available
   - Check: df['revised_eta'].notna() ? df['revised_eta'] : df['eta_dp']

10. **NULL Handling**:
    - Always check for NULL: df['column'].notna() or df['column'].isna()
    - Filter out NULLs before calculations: df[df['column'].notna()]
    - Use fillna() for missing values if appropriate

11. **Type Safety and Error Prevention** (CRITICAL):
    - ALWAYS use explicit type conversions when needed
    - String columns: df['column'].astype(str) (note: 'str' is available in execution environment)
    - Dates: pd.to_datetime(df['column'], errors='coerce')
    - Numbers: df['column'].astype(float), df['column'].astype(int)
    - Handle NaN in string conversions: df['column'].astype(str).replace('nan', '')
    - Check column existence before using: if 'column_name' in df.columns
    - Available Python builtins: str, int, float, bool, list, dict, len, min, max, sum, range
    - AVOID operations that might fail on NaN/None without proper handling

SUPPORTED DATE PARSING FUNCTIONS:
You have access to parse_time_period() function which handles:
- Month ranges: "Jun-Sep", "June to September"
- Months with year: "December 2025", "in Dec 2025"
- Single months: "December", "in Oct"
- Relative: "today", "yesterday", "next week", "last month"
- Ranges: "next 7 days", "last 30 days"

To use it: start_date, end_date, desc = parse_time_period(query)

TASK:
Generate ONLY the pandas code (no explanations, no markdown) that:
1. Processes DataFrame 'df' to answer: {query}
2. Stores final result in variable 'result'
3. Limits output to 20 rows maximum
4. Handles NULL/NaN values gracefully
5. Formats dates as strings 'YYYY-MM-DD' in output
6. Uses appropriate columns based on user terms above
7. Applies correct business logic for shipping domain

When the query is about a specific shipment (container / PO / booking / BL) or a small set of shipments, and
you build a DataFrame in 'result', PLEASE:
- Always include the main identifiers if those columns exist: 'container_number',
    'booking_number_multiple', 'po_number_multiple', 'ocean_bl_no_multiple'.
- Also include up to 2–4 of the most relevant timeline/port fields when present, such as
    'eta_dp', 'revised_eta', 'ata_dp', 'load_port', 'discharge_port', 'service_contract_number', 'job_no'.
- Keep the number of columns in 'result' small (around 4–8 max) by choosing the
    most relevant fields for the user's question.

SPECIAL HANDLING FOR JOB NUMBER QUERIES:
If the query asks for "job number" and mentions or implies a SPECIFIC container:
- FIRST filter df by that exact container_number (e.g., df[df['container_number'] == 'BMOU6674310'])
- THEN select the relevant columns including 'job_no', 'container_number', and 2-4 context fields
- DO NOT filter by only port or booking which would return multiple containers
- The result should contain ONLY the row(s) for that specific container

CODE REQUIREMENTS:
- Use ONLY: pandas (as pd), datetime, timedelta, re, numpy (as np)
- Include comments explaining logic
- Handle edge cases (empty results, null values, zero division)
- NO print statements, NO markdown blocks, NO explanatory text
- Start directly with code
- Use parse_time_period() for date range queries

EXAMPLE PATTERNS:

# Example 1: Delay calculation (MOST IMPORTANT PATTERN)
df['eta_dp'] = pd.to_datetime(df['eta_dp'], errors='coerce')
df['ata_dp'] = pd.to_datetime(df['ata_dp'], errors='coerce')
# CRITICAL: Only calculate delay for arrived containers
df['delay_days'] = (df['ata_dp'] - df['eta_dp']).dt.days
# Filter for arrived AND delayed containers
result = df[(df['ata_dp'].notna()) & (df['delay_days'] > 0)][['container_number', 'delay_days', 'discharge_port', 'ata_dp']].head(20)

# Example 2: In transit containers
df['ata_dp'] = pd.to_datetime(df['ata_dp'], errors='coerce')
# In transit = NOT arrived yet
result = df[df['ata_dp'].isna()][['container_number', 'load_port', 'discharge_port', 'eta_dp', 'revised_eta']].head(20)

# Example 3: Group by carrier with count
result = df.groupby('final_carrier_name')['container_number'].nunique().reset_index()
result.columns = ['Carrier', 'Container_Count']
result = result.sort_values('Container_Count', ascending=False).head(20)

# Example 4: Hot containers filter
hot_df = df[df['hot_container_flag'].isin(['Y', 'Yes', 'True', '1', 1, True])]
result = hot_df[['container_number', 'discharge_port', 'eta_dp', 'hot_container_flag']].head(20)

# Example 5: Safe field retrieval (e.g., seal_number for specific container)
# Always filter first, then select columns
container_df = df[df['container_number'] == 'TRHU8539467']
# Check if column exists and handle gracefully
if 'seal_number' in df.columns:
    result = container_df[['container_number', 'seal_number']]
else:
    result = container_df[['container_number']]  # Return available columns
result = result.head(20)

# Example 6: Port filtering (case-insensitive)
port_filter = df['discharge_port'].str.contains('LOS ANGELES', case=False, na=False)
result = df[port_filter][['container_number', 'discharge_port', 'eta_dp', 'ata_dp']].head(20)

# Example 7: Using date range parser
try:
    start_date, end_date, desc = parse_time_period('{query}')
    df['eta_dp'] = pd.to_datetime(df['eta_dp'], errors='coerce')
    result = df[(df['eta_dp'] >= start_date) & (df['eta_dp'] <= end_date)]
except:
    # Fallback if date parsing fails
    result = df.head(20)

# Example 8: Job number for specific container (IMPORTANT)
# When user asks "What is the job number for container BMOU6674310" or "What is its job number"
# ALWAYS filter by the exact container first
container_df = df[df['container_number'] == 'BMOU6674310']
if 'job_no' in df.columns:
    result = container_df[['container_number', 'job_no', 'booking_number_multiple', 'discharge_port', 'eta_dp']]
else:
    result = container_df[['container_number', 'booking_number_multiple']]
result = result.head(20)

Generate the pandas code now:"""

        # Generate code
        response = llm.invoke(code_generation_prompt)
        generated_code = response.content.strip()
        
        # Clean up code (remove markdown blocks if present)
        if "```python" in generated_code:
            generated_code = re.search(r"```python\n(.*?)```", generated_code, re.DOTALL).group(1)
        elif "```" in generated_code:
            generated_code = re.search(r"```\n(.*?)```", generated_code, re.DOTALL).group(1)
        
        logger.info(f"[PANDAS FALLBACK] Generated code:\n{generated_code}")
        
        # Validate generated code syntax before execution
        try:
            compile(generated_code, '<string>', 'exec')
            logger.info("[PANDAS FALLBACK] Code syntax validation: PASSED")
        except SyntaxError as e:
            logger.error(f"[PANDAS FALLBACK] Code syntax validation FAILED: {e}")
            return f"Error: Generated code has syntax errors. Please rephrase your query more clearly."
        
        # Execute the generated code safely with restricted builtins
        import numpy as np
        
        # Safe builtins - only allow essential types and functions, no file/network access
        safe_builtins = {
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'len': len,
            'range': range,
            'min': min,
            'max': max,
            'sum': sum,
            'abs': abs,
            'round': round,
            'sorted': sorted,
            'enumerate': enumerate,
            'zip': zip,
            'any': any,
            'all': all,
            'isinstance': isinstance,
            'type': type,
            # Allow imports needed by pandas internals while still blocking
            # user-generated import statements from accessing arbitrary modules.
            '__import__': __import__,
            'None': None,
            'True': True,
            'False': False,
        }
        
        # Provide execution context using a unified namespace so that
        # generated code can reliably resolve symbols like `df`, `pd`,
        # and any intermediate variables like `container_df`.
        # Using the same dict for both globals and locals avoids scoping issues.
        exec_namespace = {
            "__builtins__": safe_builtins,
            'df': df.copy(),
            'pd': pd,
            'np': np,
            'datetime': datetime,
            'timedelta': timedelta,
            're': re,
            'parse_time_period': parse_time_period,
            'result': None,
        }
        
        try:
            exec(generated_code, exec_namespace, exec_namespace)
            result = exec_namespace.get('result')
            
            if result is None:
                return "The generated code did not produce a result. Please rephrase your query."

            # Format result for display
            if isinstance(result, pd.DataFrame):
                if len(result) == 0:
                    return "No matching records found for your query."

                # Limit rows
                df_result = result.head(20).copy()

                # Convert datetime columns to 'YYYY-MM-DD' strings for safe JSON serialization
                for col in df_result.columns:
                    try:
                        if np.issubdtype(df_result[col].dtype, np.datetime64):
                            df_result[col] = pd.to_datetime(df_result[col], errors="coerce").dt.strftime("%Y-%m-%d")
                    except Exception:
                        continue

                row_count = len(df_result)
                result_str = df_result.to_string(index=False, max_rows=20)

                if return_table:
                    # Return both human-readable summary and structured rows
                    table_rows = df_result.to_dict(orient="records")
                    return {
                        "text": f"Found {row_count} record(s):\n\n{result_str}",
                        "table": table_rows,
                    }

                return f"Found {row_count} record(s):\n\n{result_str}"

            elif isinstance(result, (pd.Series)):
                text = f"Result:\n\n{result.to_string()}"
                if return_table:
                    return {"text": text, "table": []}
                return text

            else:
                text = f"Result: {result}"
                if return_table:
                    return {"text": text, "table": []}
                return text
                
        except Exception as exec_error:
            error_msg = str(exec_error)
            logger.error(f"[PANDAS FALLBACK] Code execution error: {error_msg}")
            
            # Retry with rephrased query if we haven't exceeded max retries
            if retry_count < MAX_RETRIES:
                logger.info(f"[PANDAS FALLBACK] Retrying with rephrased query (attempt {retry_count + 1}/{MAX_RETRIES})")
                
                # Automatically rephrase query with more details about the error
                rephrased_query = f"""Original query: {query}

Previous attempt failed with error: {error_msg}

Please provide a more detailed analysis. If the error was about missing columns or types, use available columns from the schema.
If the error was about data types, add proper type conversions (e.g., .astype(str), pd.to_datetime()).
If the error was about missing values, add proper null handling (e.g., .fillna(), .dropna(), .notna())."""
                
                # Recursive retry
                return pandas_code_generator_fallback(
                    rephrased_query,
                    consignee_codes,
                    retry_count + 1,
                    return_table=return_table,
                )
            else:
                # Max retries reached
                return f"Error executing analysis: {error_msg}. Unable to process query after {MAX_RETRIES + 1} attempts. Please try rephrasing your query with more specific details."
    
    except Exception as e:
        logger.error(f"[PANDAS FALLBACK] Error in pandas code generation: {e}", exc_info=True)
        return f"Unable to process your query using pandas analysis: {str(e)}"
    
    finally:
        # Clean up thread-local consignee codes
        if consignee_codes and hasattr(threading.current_thread(), "consignee_codes"):
            delattr(threading.current_thread(), "consignee_codes")
