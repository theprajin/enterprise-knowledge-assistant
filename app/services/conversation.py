"""
Conversation history management for multi-turn RAG.

Provides in-memory conversation session storage with creation, retrieval,
message appending, and cleanup. Uses LangChain message types for 
compatibility with the LangChain prompt ecosystem.
"""

import uuid
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional
from functools import lru_cache

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

logger = logging.getLogger(__name__)


@dataclass
class ConversationSession:
    """A single conversation session with message history."""

    session_id: str
    created_at: str
    messages: list[BaseMessage] = field(default_factory=list)

    def add_human_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.messages.append(HumanMessage(content=content))

    def add_ai_message(self, content: str) -> None:
        """Add an AI response to the conversation."""
        self.messages.append(AIMessage(content=content))

    def get_history_text(self, max_turns: int = 10) -> str:
        """
        Format conversation history as text for prompt injection.
        
        Args:
            max_turns: Maximum number of recent message pairs to include.
        """
        recent = self.messages[-(max_turns * 2):]
        lines = []
        for msg in recent:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize session for API response."""
        messages = []
        for msg in self.messages:
            messages.append({
                "role": "human" if isinstance(msg, HumanMessage) else "ai",
                "content": msg.content,
            })
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "message_count": len(self.messages),
            "messages": messages,
        }


class ConversationManager:
    """
    In-memory conversation session manager.
    
    Stores conversation histories keyed by session_id.
    Sessions are ephemeral (lost on restart) — suitable for a 
    portfolio demonstration project.
    """

    def __init__(self):
        self._sessions: dict[str, ConversationSession] = {}

    def create_session(self) -> ConversationSession:
        """Create a new conversation session and return it."""
        session_id = str(uuid.uuid4())
        session = ConversationSession(
            session_id=session_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._sessions[session_id] = session
        logger.info(f"Created conversation session: {session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get an existing session by ID, or None if not found."""
        return self._sessions.get(session_id)

    def add_exchange(
        self, session_id: str, user_message: str, ai_response: str
    ) -> None:
        """
        Add a complete user-AI exchange to a session.
        
        Args:
            session_id: The conversation session ID.
            user_message: The user's question.
            ai_response: The AI's answer.
            
        Raises:
            KeyError: If the session does not exist.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session '{session_id}' not found")

        session.add_human_message(user_message)
        session.add_ai_message(ai_response)

    def clear_session(self, session_id: str) -> bool:
        """
        Delete a conversation session.
        
        Returns:
            True if the session existed and was deleted, False otherwise.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Cleared conversation session: {session_id}")
            return True
        return False

    def list_sessions(self) -> list[dict]:
        """List all active sessions (summary only, no message bodies)."""
        return [
            {
                "session_id": s.session_id,
                "created_at": s.created_at,
                "message_count": len(s.messages),
            }
            for s in self._sessions.values()
        ]


@lru_cache()
def get_conversation_manager() -> ConversationManager:
    """Get the singleton conversation manager instance."""
    return ConversationManager()
