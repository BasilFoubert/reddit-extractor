from fastapi import FastAPI
from pydantic import BaseModel

from src.chat_service import agent_send_message, create_conversation

app = FastAPI()


class MessageRequest(BaseModel):
    content: str


class MessageResponse(BaseModel):
    content: str


class ConversationResponse(BaseModel):
    thread_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/conversations", response_model=ConversationResponse)
def new_conversation():
    thread_id = create_conversation()
    return ConversationResponse(thread_id=thread_id)


@app.post("/conversations/{thread_id}/messages", response_model=MessageResponse)
def post_message(thread_id: str, body: MessageRequest):
    reply = agent_send_message(thread_id, body.content)
    return MessageResponse(content=reply)
