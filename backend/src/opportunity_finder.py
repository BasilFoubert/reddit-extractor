from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """

""",
        ),
        (
            "human",
            "",
        ),
    ]
)


class OpportunityFinder:
    MODEL = "claude-haiku-4-5"

    def __init__(self):
        self.llm = init_chat_model(self.MODEL)
