import logging
import threading
from typing import Any, Dict, List

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

from config import settings

logger = logging.getLogger("shipping_chatbot")

import difflib
import re

from .prompts import COLUMN_SYNONYMS

# Load data dictionary once for the analyst
try:
    with open("docs/data_dictionary.md", "r") as f:
        DATA_DICTIONARY = f.read()
except:
    DATA_DICTIONARY = (
        "Data dictionary not found. Rely on general knowledge of shipping columns."
    )

ANALYST_PROMPT = """
You are a Lead Data Analyst at MCS Shipping. Your task is to generate Python/Pandas code to answer complex logistics questions.

DATA DICTIONARY:
{data_dictionary}

CURRENT DATE: {today}

CHAT HISTORY (Context):
{chat_history}

BUSINESS RULES:
- The dataframe is already loaded as `df`.
- The 'consignee_code_multiple' column is used for authorization.
- NEVER let code output data for consignees outside the authorized list.
- Use `pd.to_datetime` for any date comparisons not already handled in preprocessing.
- For month-only queries (e.g., 'October'), assume year 2025.
- ONLY if the user explicitly mentions 'hot', 'priority', 'urgent', or 'expedited' in the query, filter where `hot_container_flag` is True, 'Y', 'YES', or 1. Otherwise, ignore this flag.
- ALWAYS use `pd.notna()` or `pd.to_datetime()` when needed; `pd` and `np` are already available.
- If the user asks for a 'list' or 'show', ensure `result` contains a readable list or table of the relevant records.
- Return ONLY the final answer as a string by assigning it to a variable named `result`.

USER QUESTION: {question}

CONSTRAINTS:
- No need to import pandas or numpy (already available as `pd` and `np`).
- Do NOT load any files (df is already in scope).
- Ensure calculations are robust (handle NaNs).
- Your code will be executed via `exec()`.

GENERATE ONLY THE PYTHON CODE:
"""


class ShipmentAnalyst:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
            temperature=0,
        )
        self.prompt = ChatPromptTemplate.from_template(ANALYST_PROMPT)

    def analyze(self, question: str, df: pd.DataFrame) -> str:
        try:
            logger.info(f"Analyst Engine processing query: {question}")

            from .query_bank import match_query_bank

            bank_result = match_query_bank(question, df, llm=self.llm)
            if bank_result:
                return bank_result

            # Retrieve chat history from thread-local storage for context
            history = getattr(threading.current_thread(), "chat_history", [])
            history_text = ""
            if history:
                from langchain_core.messages import AIMessage, HumanMessage

                formatted_msgs = []
                for m in history:
                    role = "User" if isinstance(m, HumanMessage) else "Analyst"
                    formatted_msgs.append(f"{role}: {m.content}")
                history_text = "\n".join(formatted_msgs)

            # 1. Generate Code
            from datetime import datetime

            today_str = datetime.now().strftime("%Y-%m-%d")

            chain = self.prompt | self.llm
            response = chain.invoke(
                {
                    "data_dictionary": DATA_DICTIONARY,
                    "question": question,
                    "today": today_str,
                    "chat_history": history_text or "No previous history.",
                }
            )

            code = response.content.strip()
            # Clean markdown if LLM includes it
            if code.startswith("```python"):
                code = code.split("```python")[1].split("```")[0].strip()
            elif code.startswith("```"):
                code = code.split("```")[1].split("```")[0].strip()

            # DEBUG LOGGING: Extremely important for transparency
            logger.info("=" * 40)
            logger.info(f"ANALYST ENGINE DEBUG MODE")
            logger.info(f"QUERY: {question}")
            logger.info(f"GENERATED CODE:\n{code}")
            logger.info("=" * 40)

            # 2. Execute Code
            # Use a single dictionary for both globals and locals to ensure
            # that functions/lambdas have access to all variables.
            exec_context = {
                "df": df,
                "pd": pd,
                "result": None,
                "np": (
                    pd.np if hasattr(pd, "np") else None
                ),  # Optional: some older pandas have pd.np
            }
            # Adding standard imports to the context for robustness
            import numpy as np

            exec_context["np"] = np

            # --- SCHEMA GUARD: Validate and Repair Column Names ---
            valid_columns = df.columns.tolist()

            def resolve_col(col_name: str) -> str:
                # 1. Direct match
                if col_name in valid_columns:
                    return col_name
                # 2. Case-insensitive match
                for vc in valid_columns:
                    if vc.lower() == col_name.lower():
                        return vc
                # 3. Synonym match
                normalized = col_name.lower().replace("_", " ").strip()
                if normalized in COLUMN_SYNONYMS:
                    target = COLUMN_SYNONYMS[normalized]
                    if target in valid_columns:
                        return target
                # 4. Fuzzy match
                matches = difflib.get_close_matches(
                    col_name, valid_columns, n=1, cutoff=0.7
                )
                if matches:
                    return matches[0]
                return col_name

            # Find all strings in the code that are likely column names
            # Specifically looking for df['col'], df["col"], or items in df[['col1', 'col2']]
            # Also catch .str.contains('something') which we SHOULD NOT replace

            # Simple approach: find all single/double quoted strings
            potential_cols = re.findall(r"['\"]([a-zA-Z0-9_ ]+?)['\"]", code)
            repaired_code = code

            for p_col in set(potential_cols):
                # We only want to resolve it if it looks like a column name
                # (has letters, maybe underscores/spaces) and is NOT a valid column
                if p_col not in valid_columns:
                    # Check if it's used in a context that implies a column
                    # e.g. df[p_col], df[[..., p_col, ...]], .rename(columns={p_col: ...})
                    # For simplicity and effectiveness, we resolve it if it's a known synonym
                    # or has a very high fuzzy match score.
                    resolved = resolve_col(p_col)
                    if resolved != p_col:
                        logger.info(
                            f"SCHEMA GUARD: Repairing column '{p_col}' -> '{resolved}'"
                        )
                        # Use regex to replace only when quoted to avoid partial matches
                        repaired_code = re.sub(
                            f"(['\"]){re.escape(p_col)}(['\"])",
                            f"\\1{resolved}\\2",
                            repaired_code,
                        )

            if repaired_code != code:
                logger.info(f"REPAIRED CODE:\n{repaired_code}")
                code = repaired_code
            # -----------------------------------------------------

            exec(code, exec_context)

            result = exec_context.get(
                "result", "Code executed but no 'result' variable found."
            )

            logger.info(
                f"ANALYST ENGINE SUCCESS: Result obtained (length: {len(str(result))})"
            )
            return str(result)

        except Exception as e:
            logger.error(f"Analyst Engine failed: {e}", exc_info=True)
            # Log the code that failed if we have it
            if "code" in locals():
                logger.error(f"FAILED CODE:\n{locals()['code']}")
            return f"Error during data analysis: {str(e)}"


# Singleton instance
analyst = ShipmentAnalyst()


def unified_shipment_analyst(query: str, **kwargs) -> str:
    """
    Advanced analyst tool that can handle almost any query about counts,
    averages, delays, distributions, and trends in the shipping data.
    Input: Natural language question.
    Output: Analytical answer derived dynamic code execution.
    """
    from .tools import _df  # Import here to avoid circular dependency

    df = _df()

    if df.empty:
        return "No data available for your authorized consignee code(s)."

    return analyst.analyze(query, df)
