# agents/memory_manager.py
from langchain.memory import ConversationBufferWindowMemory
from typing import Dict, Optional
from datetime import datetime, timedelta
import threading
import logging

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


# Global memory manager instance
memory_manager = MemoryManager(max_messages=10, session_timeout_minutes=30)
