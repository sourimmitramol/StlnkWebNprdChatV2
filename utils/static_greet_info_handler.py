# utils/static_greet_info_handler.py
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from config import settings
from utils.logger import logger

_OVERVIEW_DOC_PATH = (
    Path(__file__).resolve().parents[1] / "docs" / "overview_info.md"
)

_GREETINGS = [
    "hi",
    "hello",
    "hey",
    "gm",
    "good morning",
    "good afternoon",
    "good evening",
    "hola",
]
_THANKS = ["thank", "thx", "thanks", "thank you", "ty", "much appreciated"]
_FAREWELLS = ["bye", "goodbye", "see you", "take care", "cya", "see ya", "good bye"]
_SHIPPING_KEYWORDS = [
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
_COMPANY_KEYWORDS = [
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
_OVERVIEW_PHRASES = [
    "overview",
    "what can you do",
    "what do you do",
    "what is this",
    "tell me about",
    "about you",
    "about this",
    "capabilities",
    "features",
]
_HELP_PHRASES = ["help", "help me", "need help", "support"]
_FOLLOWUP_PHRASES = [
    "tell me more",
    "more info",
    "more information",
    "details",
    "go on",
    "continue",
    "what else",
]

_DEFAULT_OVERVIEW_TEXT = (
    "MOL Consolidation Service (MCS) is a global logistics company providing "
    "supply chain solutions including ocean consolidation, freight forwarding, "
    "and more."
)
_MAX_HISTORY_MESSAGES = 10


@dataclass
class SessionState:
    greeted: bool = False
    overview_shown: bool = False
    last_intent: Optional[str] = None
    last_seen: float = 0.0
    chat_history: list = field(default_factory=list)


_sessions_lock = threading.Lock()
_sessions: dict[str, SessionState] = {}
_overview_cache: Optional[str] = None


def is_static_greet_info_enabled() -> bool:
    return bool(settings.STATIC_GREET_INFO_ENABLED)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _has_token(token_list: list[str], text: str) -> bool:
    for token in token_list:
        if " " in token:
            if token in text:
                return True
        else:
            if re.search(rf"\b{re.escape(token)}\b", text):
                return True
    return False


def _is_shipping_query(q: str) -> bool:
    return _has_token(_SHIPPING_KEYWORDS, q)


def _load_overview_text() -> str:
    try:
        text = _OVERVIEW_DOC_PATH.read_text(encoding="utf-8").strip()
        if text:
            return text
    except FileNotFoundError:
        logger.warning("Overview info file missing: %s", _OVERVIEW_DOC_PATH)
    except Exception as exc:
        logger.warning("Failed to read overview info: %s", exc)
    return _DEFAULT_OVERVIEW_TEXT


def _get_overview_text() -> str:
    global _overview_cache
    if _overview_cache is None:
        _overview_cache = _load_overview_text()
    return _overview_cache


def _get_session_ttl_seconds() -> int:
    try:
        return max(int(settings.STATIC_GREET_INFO_SESSION_TTL_SECONDS), 60)
    except Exception:
        return 1800


def _get_session_state(session_id: str) -> SessionState:
    now = time.time()
    ttl = _get_session_ttl_seconds()

    with _sessions_lock:
        if ttl > 0:
            expired = [
                sid for sid, state in _sessions.items() if now - state.last_seen > ttl
            ]
            for sid in expired:
                _sessions.pop(sid, None)

        state = _sessions.get(session_id)
        if state is None:
            state = SessionState(last_seen=now)
            _sessions[session_id] = state
        else:
            state.last_seen = now

    return state


def _build_response(text: str) -> dict:
    return {"response": text, "observation": [], "table": [], "mode": "direct"}


def _mark_state(state: SessionState, intent: str) -> None:
    state.last_intent = intent
    if intent == "greeting":
        state.greeted = True
    if intent == "overview":
        state.overview_shown = True


def _store_history(state: SessionState, user_text: str, assistant_text: str) -> None:
    state.chat_history.append(HumanMessage(content=user_text))
    state.chat_history.append(AIMessage(content=assistant_text))
    if len(state.chat_history) > _MAX_HISTORY_MESSAGES:
        state.chat_history = state.chat_history[-_MAX_HISTORY_MESSAGES:]


def _is_greeting(q: str) -> bool:
    return _has_token(_GREETINGS, q)


def _is_thanks(q: str) -> bool:
    return _has_token(_THANKS, q) and not _is_shipping_query(q)


def _is_farewell(q: str) -> bool:
    return _has_token(_FAREWELLS, q)


def _is_overview(q: str) -> bool:
    if _has_token(_OVERVIEW_PHRASES, q):
        return not _is_shipping_query(q) or q == "overview"
    if _has_token(_HELP_PHRASES, q):
        return not _is_shipping_query(q)
    return False


def _is_followup(q: str) -> bool:
    return _has_token(_FOLLOWUP_PHRASES, q) and not _is_shipping_query(q)


def _answer_company_query(query: str, chat_history: Optional[list]) -> str:
    try:
        llm = AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            temperature=0.1,
            max_tokens=1200,
        )

        system_prompt = (
            "You are MCS AI, a helpful and friendly assistant for MOL Consolidation "
            "Service (MCS). Use the following company information to answer the "
            "user's question concisely and accurately.\n\n"
            "### COMPANY INFORMATION ###\n"
            f"{_get_overview_text()}\n\n"
            "### CONSTRAINTS ###\n"
            "1. Answer based on the provided company information first.\n"
            "2. If the user asks a follow-up question, use the chat history to "
            "maintain context.\n"
            "3. If the information is not in the context above, answer neutrally "
            "that you are primarily a shipping assistant."
        )

        messages = [SystemMessage(content=system_prompt)]
        if chat_history:
            messages.extend(chat_history[-_MAX_HISTORY_MESSAGES:])
        messages.append(HumanMessage(content=query))

        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as exc:
        logger.error("Static greet handler failed: %s", exc)
        return (
            "Sorry, I couldn't process that request right now. "
            "Please try again later."
        )


def handle_static_greet_info(
    query: str, session_id: Optional[str] = None
) -> Optional[dict]:
    if not query:
        return None

    q = _normalize(query)
    session_key = (session_id or "").strip()
    state = _get_session_state(session_key) if session_key else None

    if state and _is_followup(q) and state.last_intent in {"greeting", "overview"}:
        text = _get_overview_text()
        _mark_state(state, "overview")
        _store_history(state, query, text)
        return _build_response(text)

    if _is_greeting(q):
        if state and state.greeted:
            text = "Welcome back! How can I help you today?"
        else:
            text = (
                "Hello! I'm MCS AI, An AI powered MCS Chatbot, your shipping "
                "assistant. How can I help you today?"
            )
        if state:
            _mark_state(state, "greeting")
            _store_history(state, query, text)
        return _build_response(text)

    if _is_thanks(q):
        text = "You're very welcome! Always happy to help. - MCS AI"
        if state:
            _mark_state(state, "thanks")
            _store_history(state, query, text)
        return _build_response(text)

    if _has_token(["how are you", "how r u"], q):
        text = "I'm doing great, thanks for asking! How about you? - MCS AI"
        if state:
            _mark_state(state, "smalltalk")
            _store_history(state, query, text)
        return _build_response(text)

    if _has_token(
        ["who are you", "your name", "what is your name", "what's your name"], q
    ):
        text = (
            "I'm MCS AI, your AI-powered shipping assistant. "
            "I can help you track containers, POs, and company information."
        )
        if state:
            _mark_state(state, "smalltalk")
            _store_history(state, query, text)
        return _build_response(text)

    if _is_farewell(q):
        text = "Goodbye! Have a wonderful day ahead. - MCS AI"
        if state:
            _mark_state(state, "farewell")
            _store_history(state, query, text)
        return _build_response(text)

    if _is_overview(q):
        text = _get_overview_text()
        if state:
            _mark_state(state, "overview")
            _store_history(state, query, text)
        return _build_response(text)

    is_shipping = _is_shipping_query(q)
    is_company = _has_token(_COMPANY_KEYWORDS, q)
    if (not is_shipping) or is_company:
        answer = _answer_company_query(
            query, state.chat_history if state else None
        )
        if state:
            _mark_state(state, "company")
            _store_history(state, query, answer)
        return _build_response(answer)

    return None
