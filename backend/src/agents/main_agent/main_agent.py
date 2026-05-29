from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.agents.main_agent.tools import (
    build_user_thread,
    extract_pain_points,
    filter_pain_points,
    get_cluster,
    inspect_state,
    list_clusters,
    list_tmp_files,
    spot_clusters,
)
from src.core.llm import get_llm

SYSTEM_PROMPT = (
    "You are a helpful assistant designed to conduct market research by analyzing reddit posts and explore pain points of redditors. "
    "Each response must be as concise as possible like in a casual but professional conversation. "
    "Never specify date formats or technical input requirements to the user — just ask naturally as a human would. "
    "Before calling a tool, only collect the parameters it explicitly requires — never ask the user for anything else."
    "Do not write in markdown, just write normal text without special formating."
)

POST_BUILD_PROMPT = (
    "The subreddit has just been downloaded. Recommend the next step: extracting pain points "
    "using extract_pain_points with the pickle filename returned above. "
)

POST_INSPECT_STATE_PROMPT = (
    "The pipeline state has just been inspected. Based on what is already computed, "
    "tell the user clearly what steps are done and what remains. "
    "Then recommend the next logical step and offer to run it."
)

POST_SPOT_CLUSTERS_PROMPT = (
    "The clustering just ran. Confirm to the user how many macro-clusters were found, "
    "then suggest calling list_clusters to see the full overview."
)

POST_LIST_CLUSTERS_PROMPT = (
    "The macro-clusters have just been listed. Present them as a clean numbered list: "
    "for each cluster give its description and pain point count. "
    "After the list, invite the user to explore any cluster in detail by asking which one interests them — "
    "you will then call get_cluster with its index."
)

POST_GET_CLUSTER_PROMPT = (
    "A macro-cluster's full details have just been returned. "
    "Present the pain points clearly: group the most urgent ones (8-10) first, then the rest. "
    "After listing them, give a short synthesis: what is the core frustration, who is affected, "
    "and what kind of solution or product could address it. Keep it concise."
)

INITIAL_MESSAGE = (
    "Hi! I'm your Reddit market research assistant. "
    "To identify pain points, provide me with a subreddit name to begin with. "
)

tools = [
    build_user_thread,
    list_tmp_files,
    inspect_state,
    extract_pain_points,
    filter_pain_points,
    spot_clusters,
    list_clusters,
    get_cluster,
]
llm = get_llm().bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _build_system_prompt(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        if msg.name == "inspect_state":
            return SYSTEM_PROMPT + " " + POST_INSPECT_STATE_PROMPT
        if msg.name == "spot_clusters":
            return SYSTEM_PROMPT + " " + POST_SPOT_CLUSTERS_PROMPT
        if msg.name == "list_clusters":
            return SYSTEM_PROMPT + " " + POST_LIST_CLUSTERS_PROMPT
        if msg.name == "get_cluster":
            return SYSTEM_PROMPT + " " + POST_GET_CLUSTER_PROMPT
        if msg.name == "build_user_thread":
            return SYSTEM_PROMPT + " " + POST_BUILD_PROMPT
    return SYSTEM_PROMPT


def chat_node(state: ChatState) -> ChatState:
    messages = state.get("messages", [])
    system_prompt = _build_system_prompt(messages)
    try:
        response = llm.invoke([SystemMessage(content=system_prompt)] + messages)
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
