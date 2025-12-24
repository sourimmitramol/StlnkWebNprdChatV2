# agents/azure_agent.py
import logging
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


def initialize_azure_agent(tools: List[Tool] | None = None) -> Tuple[object, AzureChatOpenAI]:
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
        temperature=0.03  # Adjust temperature for creativity (0.0 = deterministic, 1.0 = creative)
    )
    # Build a structured-chat agent prompt (offline, no LangChain Hub dependency)
    # Required variables for create_structured_chat_agent: tools, tool_names, input, agent_scratchpad
    system_instructions = (
        f"{ROBUST_COLUMN_MAPPING_PROMPT}\n\n"
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

    runnable_agent = create_structured_chat_agent(llm=llm, tools=tools or TOOLS, prompt=prompt)
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
        llm,
        db=engine,
        agent_type="openai-tools",
        verbose=True
    )
    return sql_agent_executor












