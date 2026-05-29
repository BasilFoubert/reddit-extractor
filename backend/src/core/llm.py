from langchain.chat_models import init_chat_model

MODEL = "claude-haiku-4-5"


def get_llm(**kwargs):
    return init_chat_model(MODEL, model_provider="anthropic", **kwargs)
