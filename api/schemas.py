# api/schemas.py
from typing import List, Optional

from pydantic import BaseModel, Field

# class AskRequest(BaseModel):
#     question: str = Field(
#         ..., min_length=1, description="User question for the shipping chatbot"
#     )


class QueryWithConsigneeBody(BaseModel):
    question: str
    consignee_code: str
    session_id: Optional[str] = None
