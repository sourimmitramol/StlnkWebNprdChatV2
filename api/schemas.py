# api/schemas.py
import uuid
from typing import Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, description="User question for the shipping chatbot"
    )


class QueryWithConsigneeBody(BaseModel):
    question: str
    consignee_code: str
    session_id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique session ID for conversation continuity",
    )
