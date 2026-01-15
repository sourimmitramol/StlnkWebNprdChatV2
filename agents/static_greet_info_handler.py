import logging
import os
import re

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from config import settings

logger = logging.getLogger("shipping_chatbot")

# Load company information for handling general queries related to MCS
try:
    # Use relative path from project root
    file_path = os.path.join(os.getcwd(), "docs", "overview_info.md")
    with open(file_path, "r", encoding="utf-8") as f:
        COMPANY_OVERVIEW = f.read()
except Exception as e:
    logger.error(f"Failed to load overview_info.md: {e}")
    COMPANY_OVERVIEW = "MOL Consolidation Service (MCS) is a global logistics company providing supply chain solutions including ocean consolidation, freight forwarding, and more."


def handle_non_shipping_queries(query: str, chat_history: list = None) -> str:
    """
    Handle greetings, thanks, small talk, and general MCS company related queries.
    Uses Azure OpenAI for knowledge-based questions using overview_info.md context.
    Supports chat history for follow-up questions.
    """
    q = query.lower().strip()

    # Use word boundaries for small-talk to avoid matching "hi" in "shipping"
    def has_word(word_list, text):
        for word in word_list:
            pattern = rf"\b{re.escape(word)}\b"
            if re.search(pattern, text):
                return True
        return False

    # Quick responses for small-talk
    greetings = [
        "hi",
        "hello",
        "hey",
        "gm",
        "good morning",
        "good afternoon",
        "good evening",
        "hola",
    ]
    if has_word(greetings, q):
        return "Hello! I'm MCS AI, An AI powered MCS Chatbot, your shipping assistant. How can I help you today?"

    thanks = ["thank", "thx", "thanks", "thank you", "ty", "much appreciated"]
    if has_word(thanks, q):
        return "You're very welcome! Always happy to help. - MCS AI"

    if has_word(["how are you", "how r u"], q):
        return "I'm doing great, thanks for asking! How about you? - MCS AI"

    if has_word(["who are you", "your name", "what is your name"], q):
        return "I'm MCS AI, your AI-powered shipping assistant. I can help you track containers, POs, and company information."

    farewells = ["bye", "goodbye", "see you", "take care", "cya", "see ya"]
    if has_word(farewells, q):
        return "Goodbye! Have a wonderful day ahead. - MCS AI"

    # Keywords that trigger the LLM to search the company knowledge base
    shipping_keywords = [
        "container",
        "shipment",
        "cargo",
        "po",
        "eta",
        "vessel",
        "port",
        "delivery",
        "bill of lading",
        "obl",
        "tracking",
    ]
    company_keywords = [
        "mcs",
        "mol",
        "starlink",
        "office",
        "location",
        "ceo",
        "history",
        "service",
        "mission",
        "vision",
        "value",
        "established",
        "founded",
    ]

    is_shipping = any(word in q for word in shipping_keywords)
    is_company = any(word in q for word in company_keywords)

    # If it's a company query or just a generic question, use the Knowledge Base
    if not is_shipping or is_company:
        try:
            llm = AzureChatOpenAI(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
                azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                temperature=0.1,  # Factual consistency
                max_tokens=1200,
            )

            system_prompt = (
                "You are MCS AI, a helpful and friendly assistant for MOL Consolidation Service (MCS). "
                "Use the following company information to answer the user's question concisely and accurately.\n\n"
                "### COMPANY INFORMATION ###\n"
                f"{COMPANY_OVERVIEW}\n\n"
                "### CONSTRAINTS ###\n"
                "1. Answer based ON THE PROVIDED COMPANY INFORMATION first.\n"
                "2. If the user asks a follow-up question, use the chat history to maintain context.\n"
                "3. If the information is not in the context above, answer neutrally that you are primarily a shipping assistant."
            )

            # Build messages including history
            messages = [SystemMessage(content=system_prompt)]

            if chat_history:
                # chat_history is expected to be a list of BaseMessage objects
                messages.extend(chat_history[-5:])  # Keep last 5 messages for context

            messages.append(HumanMessage(content=query))

            response = llm.invoke(messages)
            return response.content.strip()

        except Exception as e:
            logger.error(f"Static Greet Handler failed: {e}")
            return f"Sorry, I couldn't process that request right now. Please try again later. ({e})"

    # Default fallback
    return "That doesn't look like a shipping-related question, but I'm MCS AI and I'm here to help! What would you like to know?"
