# agents/azure_agent.py
import logging
from typing import List, Tuple

from langchain.agents import AgentExecutor, Tool, create_structured_chat_agent
from langchain.agents.structured_chat.prompt import FORMAT_INSTRUCTIONS, PREFIX
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
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

    # --- HARD RESTRICTION TO CORE TOOLS ---
    core_tool_names = [
        "Get Today Date",
        "Get Container Milestones",
        "Analyze Data with Pandas",
        "Handle Non-shipping queries",
    ]
    tools = [t for t in tools if t.name in core_tool_names]
    # --------------------------------------

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
        "CORE TOOL PREFERENCE RULES:\n"
        "1. For FULL Milestone History, tracking events, or detailed status timelines of a SPECIFIC ID -> use 'Get Container Milestones'.\n"
        "2. For specific fields (who is vendor, supplier name, latest ETA, discharge port, carrier name) or any and all data processing/aggregations -> ALWAYS PRIORITY GOTO: 'Analyze Data with Pandas'.\n"
        "3. For greetings, small talk, or general company info -> use 'Handle Non-shipping queries'.\n"
        "4. If 'Analyze Data with Pandas' returns no data but the query is about company info, failover to 'Handle Non-shipping queries'.\n"
        "5. Use 'Get Today Date' immediately for relative timing (today, next month, etc.) before choosing other tools.\n\n"
        "CRITICAL: You MUST respond in the structured JSON format specified below. Even your 'Final Answer' MUST be a JSON blob with action='Final Answer' and your response in 'action_input'.\n\n"
        "CONSIGNEE AUTHORIZATION CONTEXT:\n"
        "- The 'Authorized Consignee(s)' listed below are for permission context only.\n"
        "- NEVER use these codes as shipment identifiers (PO/Container/BL).\n"
        "- DO NOT include these codes in tool inputs (e.g., 'Analyze Data with Pandas'). Tools automatically filter by these codes using the system context. Only pass the actual shipment identifiers (PO/Container) asked by the user.\n\n"
        "CRITICAL DATE HANDLING RULES:\n"
        "- When the user mentions a month without a year (e.g., 'October', 'Oct', 'December'), ALWAYS assume the year is 2025\n"
        "- NEVER assume 2026 or any future year unless explicitly stated by the user\n"
        "- For all date-related queries, default to year 2025\n"
        f"{PREFIX}\n\n"
        "{tools}\n\n"
        f"{FORMAT_INSTRUCTIONS}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_instructions),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                "USER QUESTION: {input}\n\nAUTHORIZED CONSIGNEE CODES: {consignee_code}\n\n{agent_scratchpad}",
            ),
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
