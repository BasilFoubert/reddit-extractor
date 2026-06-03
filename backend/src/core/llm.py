from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(Path(__file__).parents[3] / ".env")

MODEL = "claude-haiku-4-5"


def get_llm(**kwargs):
    return init_chat_model(MODEL, model_provider="anthropic", **kwargs)
