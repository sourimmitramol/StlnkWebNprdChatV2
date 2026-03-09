# agents/memory_manager.py
import logging
import re
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional

try:
    # LangChain 0.2.x style import
    from langchain.memory import ConversationBufferWindowMemory
except ImportError:
    # LangChain 1.x compatibility fallback
    from langchain_classic.memory import ConversationBufferWindowMemory

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
                        "vessels": [],
                        "jobs": [],
                    },
                    "last_query_context": {  # Track context from last query
                        "consignee_name": None,
                        "location": None,
                        "time_period": None,
                        "carrier": None,
                        "query_type": None,  # e.g., "container_lookup", "hot_containers", etc.
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

        # Extract vessel names (common vessel name patterns)
        vessel_patterns = [
            r"\b(?:vessel|ship)[:\s]+([A-Z][A-Z\s]{3,30})\b",  # "vessel: GUDRUN MAERSK"
            r"['\"']([A-Z][A-Z\s]{3,30})['\"']",  # 'GUDRUN MAERSK'
            r"final_vessel_name['\"]?:\s*['\"]([^'\"]+)['\"]",  # JSON field
            r"\bwith\s+([A-Z]{3,}(?:\s+[A-Z]{3,})+)\b",  # "with GUDRUN MAERSK"
            r"\bassociated with\s+([A-Z]{3,}(?:\s+[A-Z]{3,})+)\b",  # "associated with GUDRUN MAERSK"
            r"\bfor\s+([A-Z]{3,}(?:\s+[A-Z]{3,})+)\b",  # "for MAERSK FORTALEZA"
        ]
        for pattern in vessel_patterns:
            vessels = re.findall(pattern, text, re.IGNORECASE)
            for vessel in vessels:
                vessel = vessel.strip().upper()
                # Ensure it looks like a vessel name (2+ words, each word 3+ chars)
                words = vessel.split()
                if (
                    len(words) >= 2
                    and all(len(w) >= 3 for w in words)
                    and vessel not in entities["vessels"]
                ):
                    entities["vessels"].append(vessel)
                    if len(entities["vessels"]) > 5:
                        entities["vessels"].pop(0)

        # Extract job numbers (pattern: 2-3 letters, 1-2 digits, 3-4 letters/digits, optional suffix)
        job_patterns = [
            r"\b((?:CN|VN|TH|HK|SG|ID|MY|PH|IN|KR|JP)\d{1,2}[A-Z]{3,4}\d{4}(?:-\d{3,4}(?:[A-Z]\d{3})?)?|[A-Z]{2}\d[A-Z]{3}\d{2}[A-Z]\d{3}[A-Z])\b",  # CN8WSG2507-0021, HK3WSG25C003M
            r"\bjob[_\s]?(?:no|number)?[:\s]*([A-Z]{2,3}\d[A-Z]{3,4}\d{4}(?:-\d{3,4})?)",  # "job_no: TH2WSG1712"
        ]
        for pattern in job_patterns:
            jobs = re.findall(pattern, text, re.IGNORECASE)
            for job in jobs:
                job = job.upper().strip()
                if job and job not in entities["jobs"] and len(job) >= 6:
                    entities["jobs"].append(job)
                    if len(entities["jobs"]) > 10:  # Keep more job numbers
                        entities["jobs"].pop(0)

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
            for entity_type in [
                "containers",
                "pos",
                "bls",
                "bookings",
                "vessels",
                "jobs",
            ]:
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

            # Resolve vessel references
            if re.search(
                r"\b(?:this|that|the)\s+(?:vessel|ship)\b", query, re.IGNORECASE
            ):
                if entities["vessels"]:
                    last_vessel = entities["vessels"][-1]
                    resolved_query = re.sub(
                        r"\b(?:this|that|the)\s+(?:vessel|ship)\b",
                        f"vessel {last_vessel}",
                        resolved_query,
                        flags=re.IGNORECASE,
                    )
                    logger.info(f"✅ Resolved 'this vessel' to '{last_vessel}'")

            # Resolve generic "this" - try to infer from context
            if (
                re.search(r"\bthis\b", query, re.IGNORECASE)
                and "associated with this" in query.lower()
            ):
                # Priority: vessel > job > container > PO
                if entities["vessels"]:
                    last_vessel = entities["vessels"][-1]
                    resolved_query = re.sub(
                        r"\bassociated with this\b",
                        f"associated with vessel {last_vessel}",
                        resolved_query,
                        flags=re.IGNORECASE,
                    )
                    logger.info(
                        f"✅ Resolved 'associated with this' to vessel '{last_vessel}'"
                    )
                elif entities["jobs"]:
                    # Use last 5 job numbers if multiple tracked
                    recent_jobs = entities["jobs"][-5:]
                    jobs_str = ", ".join(recent_jobs)
                    resolved_query = re.sub(
                        r"\bassociated with this\b",
                        f"associated with jobs {jobs_str}",
                        resolved_query,
                        flags=re.IGNORECASE,
                    )
                    logger.info(
                        f"✅ Resolved 'associated with this' to jobs '{jobs_str}'"
                    )
                elif entities["containers"]:
                    last_container = entities["containers"][-1]
                    resolved_query = re.sub(
                        r"\bassociated with this\b",
                        f"associated with container {last_container}",
                        resolved_query,
                        flags=re.IGNORECASE,
                    )
                    logger.info(
                        f"✅ Resolved 'associated with this' to container '{last_container}'"
                    )

            if resolved_query != query:
                logger.info(f"Entity resolution: '{query}' -> '{resolved_query}'")

            return resolved_query

    def update_query_context(
        self,
        session_id: str,
        consignee_name: str = None,
        location: str = None,
        time_period: str = None,
        carrier: str = None,
        query_type: str = None,
    ):
        """
        Update the last query context for a session.

        Args:
            session_id: Unique identifier for the session
            consignee_name: Consignee name mentioned in query
            location: Location/port mentioned in query
            time_period: Time period mentioned in query
            carrier: Carrier mentioned in query
            query_type: Type of query (e.g., "hot_containers", "delayed_containers")
        """
        with self.lock:
            if session_id not in self.memories:
                return

            context = self.memories[session_id]["last_query_context"]
            if consignee_name:
                context["consignee_name"] = consignee_name
            if location:
                context["location"] = location
            if time_period:
                context["time_period"] = time_period
            if carrier:
                context["carrier"] = carrier
            if query_type:
                context["query_type"] = query_type

            logger.info(f"Updated query context for {session_id}: {context}")

    def resolve_contextual_query(self, session_id: str, query: str) -> str:
        """
        Resolve vague queries like "are there any hot containers in this"
        using the last query context.

        Args:
            session_id: Unique identifier for the session
            query: User query that may need context resolution

        Returns:
            Query enriched with context from previous query
        """
        with self.lock:
            if session_id not in self.memories:
                return query

            context = self.memories[session_id]["last_query_context"]
            resolved_query = query

            # Check if query is asking about "hot containers in this/these"
            hot_in_this_pattern = (
                r"\bhot\s+containers?\s+(?:in|from|for)\s+(?:this|these|that|those)\b"
            )
            if re.search(hot_in_this_pattern, query, re.IGNORECASE):
                # Build query using previous context
                query_parts = ["hot containers"]

                if context.get("consignee_name"):
                    query_parts.append(f"for {context['consignee_name']}")
                if context.get("location"):
                    query_parts.append(f"at {context['location']}")
                if context.get("time_period"):
                    query_parts.append(f"{context['time_period']}")
                if context.get("carrier"):
                    query_parts.append(f"with {context['carrier']}")

                resolved_query = " ".join(query_parts)
                logger.info(
                    f"✅ Resolved contextual query: '{query}' -> '{resolved_query}' "
                    f"using context: {context}"
                )

            # Check for other vague "this" references
            elif re.search(r"\bin\s+(?:this|these|that|those)\b", query, re.IGNORECASE):
                # Try to add context from previous query
                if context.get("consignee_name") and "consignee" not in query.lower():
                    resolved_query = f"{query} for {context['consignee_name']}"
                    logger.info(
                        f"✅ Added consignee context: '{query}' -> '{resolved_query}'"
                    )
                elif (
                    context.get("location")
                    and "location" not in query.lower()
                    and "port" not in query.lower()
                ):
                    resolved_query = f"{query} at {context['location']}"
                    logger.info(
                        f"✅ Added location context: '{query}' -> '{resolved_query}'"
                    )

            return resolved_query

    def get_query_context(self, session_id: str) -> dict:
        """
        Get the last query context for a session.

        Args:
            session_id: Unique identifier for the session

        Returns:
            Dictionary containing last query context
        """
        with self.lock:
            if session_id in self.memories:
                return self.memories[session_id]["last_query_context"].copy()
            return {}


# Global memory manager instance
memory_manager = MemoryManager(max_messages=10, session_timeout_minutes=30)
