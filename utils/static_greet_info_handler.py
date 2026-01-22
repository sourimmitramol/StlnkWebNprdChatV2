# utils/static_greet_info_handler.py
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
    "shipped",
    "container",
    "quantity",
    "cargo",
    "po",
    "eta",
    "vessel",
    "port",
    "delivery",
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
    "I'm MCS AI, your shipping assistant. Ask me about container status, "
    "ETAs, delays, POs, and bookings."
)


@dataclass
class SessionState:
    greeted: bool = False
    overview_shown: bool = False
    last_intent: Optional[str] = None
    last_seen: float = 0.0


_sessions_lock = threading.Lock()
_sessions: dict[str, SessionState] = {}
_overview_cache: Optional[str] = None


def is_static_greet_info_enabled() -> bool:
    return bool(settings.STATIC_GREET_INFO_ENABLED)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _contains_shipping_keywords(q: str) -> bool:
    return any(keyword in q for keyword in _SHIPPING_KEYWORDS)


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


def _is_greeting(q: str) -> bool:
    return any(
        q == word or q.startswith(word + " ") or q.startswith(word + ",")
        for word in _GREETINGS
    )


def _is_thanks(q: str) -> bool:
    return any(word in q for word in _THANKS) and not _contains_shipping_keywords(q)


def _is_farewell(q: str) -> bool:
    return any(word in q for word in _FAREWELLS)


def _is_overview(q: str) -> bool:
    if any(phrase in q for phrase in _OVERVIEW_PHRASES):
        return not _contains_shipping_keywords(q) or q == "overview"
    if any(phrase == q or phrase in q for phrase in _HELP_PHRASES):
        return not _contains_shipping_keywords(q)
    return False


def _is_followup(q: str) -> bool:
    if any(phrase == q or phrase in q for phrase in _FOLLOWUP_PHRASES):
        return not _contains_shipping_keywords(q)
    return False


def handle_static_greet_info(
    query: str, session_id: Optional[str] = None
) -> Optional[dict]:
    if not query:
        return None

    q = _normalize(query)
    session_key = (session_id or "").strip()
    state = _get_session_state(session_key) if session_key else None

    if state and _is_followup(q) and state.last_intent in {"greeting", "overview"}:
        _mark_state(state, "overview")
        return _build_response(_get_overview_text())

    if _is_greeting(q):
        if state and state.greeted:
            text = "Welcome back! How can I help you today?"
        else:
            text = (
                "Hello! I'm MCS AI, your shipping assistant. "
                "How can I help you today?"
            )
        if state:
            _mark_state(state, "greeting")
        return _build_response(text)

    if _is_thanks(q):
        if state:
            _mark_state(state, "thanks")
        return _build_response("You're very welcome! Always happy to help. - MCS AI")

    if "how are you" in q or "how r u" in q or "how are u" in q:
        if state:
            _mark_state(state, "smalltalk")
        return _build_response(
            "I'm doing great, thanks for asking! "
            "How can I assist you with your shipping needs today?"
        )

    if (
        "who are you" in q
        or "your name" in q
        or "what is your name" in q
        or "what's your name" in q
    ):
        if state:
            _mark_state(state, "smalltalk")
        return _build_response(
            "I'm MCS AI, your AI-powered shipping assistant. "
            "I can help you track containers, POs, and more."
        )

    if _is_farewell(q):
        if state:
            _mark_state(state, "farewell")
        return _build_response("Goodbye! Have a wonderful day ahead. - MCS AI")

    if _is_overview(q):
        if state:
            _mark_state(state, "overview")
        return _build_response(_get_overview_text())

    return None
