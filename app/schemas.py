from pydantic import BaseModel


class AskRequest(BaseModel):
    message: str


class AskResponse(BaseModel):
    answer: str
    intent: str
    contexts: list[dict]


class ChatPayload(BaseModel):
    message: str
