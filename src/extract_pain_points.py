from __future__ import annotations

from collections import defaultdict
from operator import add
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, Send
from tqdm import tqdm

from src.types import Comment, PainPoint, PainSummary, PostPainSummary
from src.utils import load_jsonl, save_jsonl

thread_scan_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You receive a Reddit post and all its comments.
Identify every distinct pain point or need expressed anywhere in the thread (post + comments), implicitly or explicitly.
Deduplicate: if the same pain appears multiple times, list it only once.
For each pain point assign a unique sequential index starting at 1 and write a description of 10 words maximum.""",
        ),
        (
            "human",
            "Title: {post_title}\n\nDescription: {post_descr}\n\n---\n\nComments:\n{comments}",
        ),
    ]
)


pain_dedup_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You receive a numbered list of pain points identified from a Reddit thread.
Remove duplicates: two entries are duplicates if they describe the same underlying problem or need, even if worded differently.
When merging duplicates, keep the most informative description.
Re-index the remaining entries sequentially starting at 1.
Return only the deduplicated list.""",
        ),
        ("human", "{pain_summaries}"),
    ]
)

pain_extractor_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You receive a Reddit thread (title, post body, and all comments) and ONE specific pain point or need to locate.
Search the entire thread — post title, post body, and all comments — for where this pain point is expressed, implicitly or explicitly.
Return exactly one result:
- verbatim: exact word-for-word quote from whichever part of the thread best expresses this pain
- reformulation: a self-contained, precise sentence in English that captures WHO is affected, WHAT the problem or need is, WHY it matters, and any relevant context (role, tool, constraint, consequence). Naturally weave in the author's own significant terms. Must be fully understandable without reading the original. No vague generalities — include the specific situation, domain, and stakes.
- urgency: integer from 1 to 10 reflecting how urgently a solution is needed.
  1 = mild or hypothetical concern.
  10 = actively seeking a solution right now, strong pain, or critical situation.
Do not invent anything: the verbatim must exist as-is in the thread.""",
        ),
        (
            "human",
            "Pain point to locate: {pain_description}\n\n---\n\nTitle: {post_title}\n\nPost body: {post_descr}\n\nComments:\n{comments}",
        ),
    ]
)


class PainWorkerState(TypedDict):
    post_id: str
    post_title: str
    post_descr: str
    comments_text: str
    pain_summary: PainSummary


class State(TypedDict):
    post_id: str
    post_title: str
    post_descr: str
    comments: list[Comment]
    post_pain_summary: PostPainSummary | None
    pain_points: Annotated[list[PainPoint], add]


class States(TypedDict):
    states_list: list[State]
    pain_points: Annotated[list[PainPoint], add]


class Workflow:
    MODEL = "claude-haiku-4-5"

    def __init__(self):
        self.llm = init_chat_model(self.MODEL)
        self.thread_scan_pipe = thread_scan_prompt | self.llm.with_structured_output(
            PostPainSummary
        )
        self.pain_dedup_pipe = pain_dedup_prompt | self.llm.with_structured_output(PostPainSummary)
        self.pain_extractor_pipe = pain_extractor_prompt | self.llm.with_structured_output(
            PainPoint
        )
        self.states: list[State] = self.build_states()

    @staticmethod
    def _flatten_comments_text(comments: list[Comment] | None, depth: int = 0) -> str:
        lines = []
        for c in comments or []:
            lines.append("  " * depth + "- " + c["text"])
            lines.extend(
                Workflow._flatten_comments_text(c.get("sub_comments"), depth + 1).splitlines()
            )
        return "\n".join(lines)

    def thread_scanner(self, state: State) -> dict:
        try:
            response = self.thread_scan_pipe.invoke(
                {
                    "post_title": state["post_title"],
                    "post_descr": state["post_descr"],
                    "comments": self._flatten_comments_text(state["comments"]),
                }
            )
            return {"post_pain_summary": response}
        except Exception as e:
            if "rate_limit_error" in str(e):
                raise
            return {"post_pain_summary": PostPainSummary(pain_summaries=[])}

    def pain_deduplicator(self, state: State) -> dict:
        summary = state.get("post_pain_summary")
        if not summary or len(summary.pain_summaries) <= 1:
            return {}
        formatted = "\n".join(f"{ps.index}. {ps.description}" for ps in summary.pain_summaries)
        try:
            response = self.pain_dedup_pipe.invoke({"pain_summaries": formatted})
            return {"post_pain_summary": response}
        except Exception as e:
            if "rate_limit_error" in str(e):
                raise
            return {}

    @staticmethod
    def spawn_pain_workers(state: State) -> list[Send]:
        comments_text = Workflow._flatten_comments_text(state["comments"])
        return [
            Send(
                "pain_point_extractor",
                {
                    "post_id": state["post_id"],
                    "post_title": state["post_title"],
                    "post_descr": state["post_descr"],
                    "comments_text": comments_text,
                    "pain_summary": ps,
                },
            )
            for ps in (
                state["post_pain_summary"].pain_summaries if state.get("post_pain_summary") else []
            )
        ]

    def pain_point_extractor(self, state: PainWorkerState) -> dict:
        try:
            response = self.pain_extractor_pipe.invoke(
                {
                    "pain_description": state["pain_summary"].description,
                    "post_title": state["post_title"],
                    "post_descr": state["post_descr"],
                    "comments": state["comments_text"],
                }
            )
            return {
                "pain_points": [
                    PainPoint(
                        post_id=state["post_id"],
                        verbatim=response.verbatim,
                        pain_point_reformulated=response.pain_point_reformulated,
                        urgency=response.urgency,
                    )
                ]
            }
        except Exception as e:
            if "rate_limit_error" in str(e):
                raise
            return {"pain_points": []}

    @staticmethod
    def build_states() -> list[State]:
        def _map_comment(c: dict) -> Comment:
            return {
                "text": c["body"],
                "sub_comments": [_map_comment(r) for r in c.get("replies", [])],
            }

        return [
            {
                "post_id": p["id"],
                "post_title": p["title"],
                "post_descr": p.get("selftext", ""),
                "comments": [_map_comment(c) for c in p.get("comments", [])],
                "pain_points": [],
            }
            for p in load_jsonl(THREADS_PATH)
        ]

    @staticmethod
    def spawn_post_workers(state: States) -> list[Send]:
        return [Send("process_post", s) for s in state["states_list"]]

    def build_graph(self):
        retry_policy = RetryPolicy(
            initial_interval=2.0,
            backoff_factor=2.0,
            max_interval=60,
            max_attempts=float("inf"),
            retry_on=lambda e: "rate_limit_error" in str(e),
        )

        # Inner graph: one post → scan thread → parallel pain point extractors
        post_workflow = StateGraph(State)
        post_workflow.add_node("thread_scanner", self.thread_scanner, retry=retry_policy)
        post_workflow.add_node("pain_deduplicator", self.pain_deduplicator, retry=retry_policy)
        post_workflow.add_node(
            "pain_point_extractor", self.pain_point_extractor, retry=retry_policy
        )
        post_workflow.add_edge(START, "thread_scanner")
        post_workflow.add_edge("thread_scanner", "pain_deduplicator")
        post_workflow.add_conditional_edges("pain_deduplicator", self.spawn_pain_workers)
        post_workflow.add_edge("pain_point_extractor", END)
        post_graph = post_workflow.compile()

        # Outer graph: parallel posts
        outer_workflow = StateGraph(States)
        outer_workflow.add_node("process_post", post_graph)
        outer_workflow.add_conditional_edges(START, self.spawn_post_workers)
        outer_workflow.add_edge("process_post", END)
        self.built_graph = outer_workflow.compile()


if __name__ == "__main__":
    load_dotenv()

    # THREADS_PATH = Path("data/processed/r_ciso_threads.jsonl")
    # THREADS_PATH_TEST = Path("data/processed/r_ciso_pain_points.jsonl")
    THREADS_PATH = Path("tests/data/small_subreddit.jsonl")
    THREADS_PATH_TEST = Path("tests/data/small_subreddit_pain_points.jsonl")
    wf = Workflow()
    wf.build_graph()
    all_pain_points: list[PainPoint] = []

    with tqdm(total=len(wf.states), desc="Analyzing posts", unit="post") as pbar:
        for event in wf.built_graph.stream(
            {"states_list": wf.states, "pain_points": []},
            stream_mode="updates",
        ):
            if "process_post" in event:
                all_pain_points.extend(event["process_post"].get("pain_points", []))
                pbar.update(1)

    grouped = defaultdict(list)
    for pp in all_pain_points:
        grouped[pp.post_id].append(pp)

    raw = load_jsonl(THREADS_PATH)
    enriched = [
        {**post, "pain_points": [pp.model_dump() for pp in grouped.get(post["id"], [])]}
        for post in raw
    ]
    save_jsonl(enriched, THREADS_PATH_TEST)
