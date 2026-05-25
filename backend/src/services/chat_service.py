import uuid

from src.agents.agent import _agent_send_message


def create_conversation() -> str:
    return str(uuid.uuid4())


def agent_send_message(thread_id: str, text: str) -> str:
    return _agent_send_message(text, thread_id=thread_id)
