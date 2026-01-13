# api/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question for the shipping chatbot")


class QueryWithConsigneeBody(BaseModel):
    question: str
    consignee_code: str
    session_id: Optional[str] = None
