from pydantic import BaseModel


class MessageTurn(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    message: str
    history: list[MessageTurn] = []


class AskResponse(BaseModel):
    answer: str
    intent: str
    node: str
    path: list[str]
    contexts: list[dict]


class ChatPayload(BaseModel):
    message: str
    history: list[MessageTurn] = []
