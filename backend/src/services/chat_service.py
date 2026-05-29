import uuid

from src.agents.main_agent.main_agent import INITIAL_MESSAGE, _agent_send_message


def create_conversation() -> tuple[str, str]:
    return str(uuid.uuid4()), INITIAL_MESSAGE


def agent_send_message(thread_id: str, text: str) -> str:
    return _agent_send_message(text, thread_id=thread_id)
