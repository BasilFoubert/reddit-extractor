from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.agents.tools import build_user_thread

SYSTEM_PROMPT = (
    "You are a helpful assistant designed to conduct market research by analyzing reddit posts and explore pain points of redditors. "
    "Each response must be as concise as possible like in a casual but professional conversation. "
    "Never specify date formats or technical input requirements to the user — just ask naturally as a human would. "
    "Before calling a tool, only collect the parameters it explicitly requires — never ask the user for anything else."
    "Do not write in markdown, just write normal text without special formating."
)

INITIAL_MESSAGE = (
    "Hi! I'm your Reddit market research assistant. "
    "To identify pain points, provide me with a subreddit name to begin with. "
)

tools = [build_user_thread]
llm = init_chat_model("claude-haiku-4-5", model_provider="anthropic").bind_tools(tools)


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


def _should_use_tools(state: ChatState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


workflow = StateGraph(ChatState)
workflow.add_node("agent", chat_node)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", _should_use_tools)
workflow.add_edge("tools", "agent")
graph = workflow.compile(checkpointer=MemorySaver())


def _agent_send_message(text: str, thread_id: str = "default") -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [HumanMessage(content=text)]}, config=config)
    return result["messages"][-1].content
