from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

SYSTEM_PROMPT = "You are a helpful assistant designed to conduct market research by analyzing reddit posts and explore pain points of redditors."

llm = init_chat_model("claude-haiku-4-5", model_provider="anthropic")


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState) -> ChatState:
    messages = state.get("messages", [])
    try:
        response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + messages)
        return {"messages": [response]}
    except Exception as e:
        error_message = HumanMessage(content=f"Sorry, I encountered an error: {str(e)}")
        return {"messages": [error_message]}


workflow = StateGraph(ChatState)
workflow.add_node("agent", chat_node)
workflow.add_edge("agent", END)
workflow.set_entry_point("agent")
graph = workflow.compile(checkpointer=MemorySaver())


def _agent_send_message(text: str, thread_id: str = "default") -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [HumanMessage(content=text)]}, config=config)
    return result["messages"][-1].content
