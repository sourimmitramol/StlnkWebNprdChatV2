# agents/memory_manager.py
import logging
import re
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from langchain.memory import ConversationBufferWindowMemory

logger = logging.getLogger("shipping_chatbot")


class MemoryManager:
    """Manages conversation memory for multiple sessions"""

    def __init__(self, max_messages: int = 10, session_timeout_minutes: int = 30):
        """
        Initialize the memory manager.

        Args:
            max_messages: Maximum number of message pairs to keep in memory (default: 10)
            session_timeout_minutes: Session expiry time in minutes (default: 30)
        """
        self.memories: Dict[str, Dict] = {}
        self.max_messages = max_messages
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.lock = threading.Lock()
        logger.info(
            f"MemoryManager initialized: max_messages={max_messages}, timeout={session_timeout_minutes}min"
        )

    def get_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        """
        Get or create memory for a session.

        Args:
            session_id: Unique identifier for the session

        Returns:
            ConversationBufferWindowMemory instance for the session
        """
        with self.lock:
            # Clean up expired sessions
            self._cleanup_expired_sessions()

            # Get or create memory for this session
            if session_id not in self.memories:
                logger.info(f"Creating new memory for session: {session_id}")
                self.memories[session_id] = {
                    "memory": ConversationBufferWindowMemory(
                        k=self.max_messages,
                        memory_key="chat_history",
                        return_messages=True,
                        input_key="input",
                        output_key="output",
                    ),
                    "last_accessed": datetime.now(),
                    "consignee_codes": [],
                    "created_at": datetime.now(),
                    "entities": {  # Track entities mentioned in conversation
                        "containers": [],
                        "pos": [],
                        "bls": [],
                        "bookings": [],
                    },
                }
            else:
                # Update last accessed time
                self.memories[session_id]["last_accessed"] = datetime.now()
                logger.debug(f"Retrieved existing memory for session: {session_id}")

            return self.memories[session_id]["memory"]

    def set_consignee_context(self, session_id: str, consignee_codes: list):
        """
        Store consignee codes for the session.

        Args:
            session_id: Unique identifier for the session
            consignee_codes: List of authorized consignee codes
        """
        with self.lock:
            if session_id in self.memories:
                self.memories[session_id]["consignee_codes"] = consignee_codes
                logger.debug(
                    f"Updated consignee context for session {session_id}: {consignee_codes}"
                )

    def get_consignee_context(self, session_id: str) -> list:
        """
        Retrieve consignee codes for the session.

        Args:
            session_id: Unique identifier for the session

        Returns:
            List of authorized consignee codes
        """
        with self.lock:
            if session_id in self.memories:
                return self.memories[session_id]["consignee_codes"]
            return []

    def clear_session(self, session_id: str) -> bool:
        """
        Clear memory for a specific session.

        Args:
            session_id: Unique identifier for the session

        Returns:
            True if session was cleared, False if session didn't exist
        """
        with self.lock:
            if session_id in self.memories:
                del self.memories[session_id]
                logger.info(f"Cleared session: {session_id}")
                return True
            logger.warning(f"Attempted to clear non-existent session: {session_id}")
            return False

    def _cleanup_expired_sessions(self):
        """Remove sessions that haven't been accessed recently (internal use)"""
        current_time = datetime.now()
        expired_sessions = [
            session_id
            for session_id, data in self.memories.items()
            if current_time - data["last_accessed"] > self.session_timeout
        ]

        for session_id in expired_sessions:
            del self.memories[session_id]
            logger.info(f"Expired and removed session: {session_id}")

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired session(s)")

    def get_active_sessions_count(self) -> int:
        """
        Get count of active sessions.

        Returns:
            Number of active sessions
        """
        with self.lock:
            return len(self.memories)

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """
        Get information about a specific session.

        Args:
            session_id: Unique identifier for the session

        Returns:
            Dictionary with session information or None if not found
        """
        with self.lock:
            if session_id in self.memories:
                data = self.memories[session_id]
                return {
                    "session_id": session_id,
                    "created_at": data["created_at"].isoformat(),
                    "last_accessed": data["last_accessed"].isoformat(),
                    "consignee_codes": data["consignee_codes"],
                    "message_count": len(
                        data["memory"].load_memory_variables({}).get("chat_history", [])
                    ),
                }
            return None

    def _extract_entities_internal(self, entities: Dict[str, List[str]], text: str):
        """
        Internal helper to extract entities without locking.

        Args:
            entities: Entity dictionary to update
            text: Text to extract entities from
        """
        # Extract container numbers (4 letters + 7 digits)
        containers = re.findall(r"\b[A-Z]{4}\d{7}\b", text.upper())
        for container in containers:
            if container not in entities["containers"]:
                entities["containers"].append(container)
                # Keep only last 5 containers
                if len(entities["containers"]) > 5:
                    entities["containers"].pop(0)

        # Extract PO numbers (7-13 digits, often with PO prefix)
        po_patterns = [
            r"\bPO[#\s]*([\d]{7,13})\b",
            r"\bpurchase order[#\s]*([\d]{7,13})\b",
            r"(?<!\w)([\d]{10,13})(?!\w)",  # Standalone 10-13 digit numbers
        ]
        for pattern in po_patterns:
            pos = re.findall(pattern, text, re.IGNORECASE)
            for po in pos:
                if po not in entities["pos"]:
                    entities["pos"].append(po)
                    if len(entities["pos"]) > 5:
                        entities["pos"].pop(0)

        # Extract Ocean BL numbers (various patterns)
        bl_patterns = [
            r"\b(?:OBL|ocean bl|bill of lading)[:\s]+([A-Z0-9]{10,20})\b",
            r"\b([A-Z]{3,5}[A-Z0-9]{10,15})\b",  # BL-like patterns
        ]
        for pattern in bl_patterns:
            bls = re.findall(pattern, text, re.IGNORECASE)
            for bl in bls:
                if bl not in entities["bls"] and len(bl) >= 10:
                    entities["bls"].append(bl)
                    if len(entities["bls"]) > 5:
                        entities["bls"].pop(0)

        # Extract booking numbers (6-20 alphanumeric, not container format)
        booking_patterns = [
            r"\b(?:booking|BKG)[:\s]+([A-Z]{2}\d{7,10})\b",
            r"\b([A-Z]{2}\d{7,10})\b",  # Booking-like patterns
        ]
        for pattern in booking_patterns:
            bookings = re.findall(pattern, text, re.IGNORECASE)
            for booking in bookings:
                # Avoid container numbers (AAAA#######)
                if (
                    not re.match(r"^[A-Z]{4}\d{7}$", booking)
                    and booking not in entities["bookings"]
                ):
                    entities["bookings"].append(booking)
                    if len(entities["bookings"]) > 5:
                        entities["bookings"].pop(0)

    def extract_and_track_entities(self, session_id: str, text: str):
        """
        Extract and track entities (containers, POs, BLs, bookings) from text.

        Args:
            session_id: Unique identifier for the session
            text: Text to extract entities from (questions and responses)
        """
        with self.lock:
            if session_id not in self.memories:
                return

            entities = self.memories[session_id]["entities"]
            old_count = {k: len(v) for k, v in entities.items()}
            self._extract_entities_internal(entities, text)
            new_count = {k: len(v) for k, v in entities.items()}

            # Log what was added
            added = []
            for entity_type in ["containers", "pos", "bls", "bookings"]:
                if new_count[entity_type] > old_count[entity_type]:
                    new_items = entities[entity_type][old_count[entity_type] :]
                    added.append(f"{entity_type}: {new_items}")

            if added:
                logger.info(
                    f"📝 Tracked new entities for session {session_id}: {', '.join(added)}"
                )

            logger.debug(f"Total entities for session {session_id}: {entities}")

    def resolve_entity_references(self, session_id: str, query: str) -> str:
        """
        Resolve entity references (this container, this PO, etc.) in query using tracked entities.

        Args:
            session_id: Unique identifier for the session
            query: User query that may contain references

        Returns:
            Query with resolved entity references
        """
        with self.lock:
            if session_id not in self.memories:
                return query

            entities = self.memories[session_id]["entities"]

            # CRITICAL: If entities are empty, try to extract from chat history
            # This handles cases where entity tracking was just enabled
            if not any(entities.values()):
                memory = self.memories[session_id]["memory"]
                chat_history = memory.load_memory_variables({}).get("chat_history", [])
                for msg in chat_history:
                    if hasattr(msg, "content"):
                        content = msg.content
                    elif isinstance(msg, tuple) and len(msg) >= 2:
                        content = msg[1]
                    else:
                        continue
                    # Extract entities from historical messages
                    self._extract_entities_internal(entities, content)
                logger.info(
                    f"Extracted entities from chat history for session {session_id}: {entities}"
                )

            resolved_query = query

            # Resolve container references
            if re.search(r"\b(?:this|that|the)\s+container\b", query, re.IGNORECASE):
                if entities["containers"]:
                    last_container = entities["containers"][-1]
                    resolved_query = re.sub(
                        r"\b(?:this|that|the)\s+container\b",
                        f"container {last_container}",
                        resolved_query,
                        flags=re.IGNORECASE,
                    )
                    logger.info(
                        f"✅ Resolved 'this container' to '{last_container}' (from {len(entities['containers'])} tracked containers)"
                    )
                else:
                    logger.warning(
                        f"⚠️ Query contains 'this container' but no containers tracked in session {session_id}"
                    )

            # Resolve PO references
            if re.search(
                r"\b(?:this|that|the)\s+(?:PO|purchase order)\b", query, re.IGNORECASE
            ):
                if entities["pos"]:
                    last_po = entities["pos"][-1]
                    resolved_query = re.sub(
                        r"\b(?:this|that|the)\s+(?:PO|purchase order)\b",
                        f"PO {last_po}",
                        resolved_query,
                        flags=re.IGNORECASE,
                    )
                    logger.info(f"Resolved 'this PO' to '{last_po}'")

            # Resolve BL references
            if re.search(
                r"\b(?:this|that|the)\s+(?:BL|ocean bl|bill of lading)\b",
                query,
                re.IGNORECASE,
            ):
                if entities["bls"]:
                    last_bl = entities["bls"][-1]
                    resolved_query = re.sub(
                        r"\b(?:this|that|the)\s+(?:BL|ocean bl|bill of lading)\b",
                        f"BL {last_bl}",
                        resolved_query,
                        flags=re.IGNORECASE,
                    )
                    logger.info(f"Resolved 'this BL' to '{last_bl}'")

            # Resolve booking references
            if re.search(r"\b(?:this|that|the)\s+booking\b", query, re.IGNORECASE):
                if entities["bookings"]:
                    last_booking = entities["bookings"][-1]
                    resolved_query = re.sub(
                        r"\b(?:this|that|the)\s+booking\b",
                        f"booking {last_booking}",
                        resolved_query,
                        flags=re.IGNORECASE,
                    )
                    logger.info(f"Resolved 'this booking' to '{last_booking}'")

            # Resolve plural container references ("these containers")
            if re.search(r"\b(?:these|those|the)\s+containers\b", query, re.IGNORECASE):
                if entities["containers"]:
                    # Take last 2-3 containers
                    recent_containers = entities["containers"][-3:]
                    containers_str = ", ".join(recent_containers)
                    resolved_query = re.sub(
                        r"\b(?:these|those|the)\s+containers\b",
                        f"containers {containers_str}",
                        resolved_query,
                        flags=re.IGNORECASE,
                    )
                    logger.info(f"Resolved 'these containers' to '{containers_str}'")

            if resolved_query != query:
                logger.info(f"Entity resolution: '{query}' -> '{resolved_query}'")

            return resolved_query


# Global memory manager instance
memory_manager = MemoryManager(max_messages=10, session_timeout_minutes=30)
