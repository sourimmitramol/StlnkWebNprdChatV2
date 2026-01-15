import logging
import threading
from typing import Any, Dict, List

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

from config import settings

logger = logging.getLogger("shipping_chatbot")

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

BUSINESS RULES:
- The dataframe is already loaded as `df`.
- The 'consignee_code_multiple' column is used for authorization.
- NEVER let code output data for consignees outside the authorized list.
- Use `pd.to_datetime` for any date comparisons not already handled in preprocessing.
- For month-only queries (e.g., 'October'), assume year 2025.
- Return ONLY the final answer as a string by assigning it to a variable named `result`.

USER QUESTION: {question}

CONSTRAINTS:
- Do NOT import pandas (already imported as pd).
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
            # 1. Generate Code
            chain = self.prompt | self.llm
            response = chain.invoke(
                {"data_dictionary": DATA_DICTIONARY, "question": question}
            )

            code = response.content.strip()
            # Clean markdown if LLM includes it
            if code.startswith("```python"):
                code = code.split("```python")[1].split("```")[0].strip()
            elif code.startswith("```"):
                code = code.split("```")[1].split("```")[0].strip()

            logger.info(f"Analyst Engine generated code:\n{code}")

            # 2. Execute Code
            local_vars = {"df": df, "pd": pd, "result": None}
            exec(code, {}, local_vars)

            result = local_vars.get(
                "result", "Code executed but no 'result' variable found."
            )
            return str(result)

        except Exception as e:
            logger.error(f"Analyst Engine failed: {e}", exc_info=True)
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
