from fastapi import APIRouter
from pydantic import BaseModel

from src.services.chat_service import agent_send_message, create_conversation

router = APIRouter()


class MessageRequest(BaseModel):
    content: str


class MessageResponse(BaseModel):
    content: str


class ConversationResponse(BaseModel):
    thread_id: str
    initial_message: str


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/conversations", response_model=ConversationResponse)
def new_conversation():
    thread_id, initial_message = create_conversation()
    return ConversationResponse(thread_id=thread_id, initial_message=initial_message)


@router.post("/conversations/{thread_id}/messages", response_model=MessageResponse)
def post_message(thread_id: str, body: MessageRequest):
    reply = agent_send_message(thread_id, body.content)
    return MessageResponse(content=reply)
