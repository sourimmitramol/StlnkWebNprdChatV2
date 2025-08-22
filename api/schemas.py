# api/schemas.py
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question for the shipping chatbot")


class QueryWithConsigneeBody(BaseModel):
    query: str
    consignee_codes: str
