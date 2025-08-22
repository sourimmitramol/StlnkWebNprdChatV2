# agents/azure_agent.py
import logging
from typing import List, Tuple

from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.chat_models import AzureChatOpenAI
from langchain_community.embeddings import AzureOpenAIEmbeddings
from agents.prompts import ROBUST_COLUMN_MAPPING_PROMPT

from config import settings
from .tools import TOOLS

logger = logging.getLogger("shipping_chatbot")


def initialize_azure_agent(tools: List[Tool] | None = None) -> Tuple[object, AzureChatOpenAI]:
    """Build the Zero‑Shot React agent with Azure OpenAI."""
    if tools is None:
        tools = TOOLS

    llm = AzureChatOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
    )
    # Pass system prompt as a message if needed
    system_message = {"role": "system", "content": ROBUST_COLUMN_MAPPING_PROMPT}
    # ...initialize your agent with system_message if supported...
    # For LangChain, you may need to use a custom prompt template or set system_message in the chain/tools
    agent = initialize_agent(
        tools or TOOLS,
        llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,  # <-- Add this argument
        # If using custom prompt, set it here
        # prompt=CustomPromptTemplate.from_messages([system_message])
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
