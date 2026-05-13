from __future__ import annotations
import json
from operator import add
from pathlib import Path
from typing import TypedDict, Optional, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, RetryPolicy
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from pydantic import BaseModel, Field, field_validator
from langchain_core.prompts import ChatPromptTemplate
from src.utils import load_jsonl, save_jsonl
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv





post_verbatim_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You receive the title and description of a Reddit post.
Identify every pain or need expressed by the author, implicitly or explicitly.
For each one:
- verbatim: exact word-for-word quote from the text
- reformulation: a self-contained, precise sentence in English that captures WHO is affected, WHAT the problem or need is, WHY it matters, and any relevant context (role, tool, constraint, consequence). Naturally weave in the author's own significant terms (technical names, product names, domain vocabulary, emotionally charged words) so the reformulation preserves their exact language where it matters. Must be fully understandable without reading the original post. No vague generalities — include the specific situation, domain, and stakes.
- urgency: integer from 1 to 10 reflecting how urgently the author needs a solution.
  1 = mild or hypothetical concern, no immediate action needed.
  10 = actively seeking a solution right now, strong pain, or critical situation requiring immediate resolution.
  Base this score on signals like: explicit urgency language ("asap", "critical", "stuck", "nothing works"), emotional tone, business or security risk, and whether they are actively asking for help.
Do not invent anything: the verbatim must exist as-is in the text.""",
        ),
        ("human", "Title: {post_title}\n\nDescription: {post_descr}"),
    ]
)


comment_verbatim_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You receive a Reddit post (context only) and a comment.
Identify every pain or need expressed in the COMMENT ONLY, implicitly or explicitly.
For each one:
- verbatim: exact word-for-word quote from the comment
- reformulation: a self-contained, precise sentence in English that captures WHO is affected, WHAT the problem or need is, WHY it matters, and any relevant context (role, tool, constraint, consequence) drawn from both the post and the comment. Naturally weave in the commenter's own significant terms (technical names, product names, domain vocabulary, emotionally charged words) so the reformulation preserves their exact language where it matters. Must be fully understandable without reading the originals. No vague generalities — include the specific situation, domain, and stakes.
- urgency: integer from 1 to 10 reflecting how urgently the commenter needs a solution.
  1 = mild or hypothetical concern, no immediate action needed.
  10 = actively seeking a solution right now, strong pain, or critical situation requiring immediate resolution.
  Base this score on signals like: explicit urgency language ("asap", "critical", "stuck", "nothing works"), emotional tone, business or security risk, and whether they are actively asking for help.
Do not invent anything: the verbatim must exist as-is in the comment.""",
        ),
        (
            "human",
            "Context (post, do not extract from) :\nTitle: {post_title}\nDescription: {post_descr}\n\n---\n\nComment:\n{comment}",
        ),
    ]
)



class PainPoint(BaseModel):
    post_id: str = ""
    verbatim: str = Field(description="Exact quote of the pain point from the source text")
    pain_point_reformulated: str = Field(description="Precise, self-contained reformulation including who, what, why, and context")
    urgency: int = Field(default=5, ge=1, le=10, description="Urgency level from 1 (vague or hypothetical need) to 10 (actively seeking an immediate solution, critical pain)")


class PainPoints(BaseModel):
    pain_points: list[PainPoint]

    @field_validator("pain_points", mode="before")
    @classmethod
    def parse_if_string(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return []
        return v


class Comment(TypedDict):
    text: str
    sub_comments: Optional[list[Comment]]


class CommentWorkerState(TypedDict):
    post_id: str
    post_title: str
    post_descr: str
    comment: Comment


class State(TypedDict):
    post_id: str
    post_title: str
    post_descr: str
    comments: list[Comment]
    pain_points: Annotated[list[PainPoint], add]


class States(TypedDict):
    states_list: list[State]
    pain_points: Annotated[list[PainPoint], add]




class Workflow:
    MODEL = "claude-haiku-4-5"

    def __init__(self):
        self.llm = init_chat_model(self.MODEL)
        self.post_verbatim_pipe = post_verbatim_prompt | self.llm.with_structured_output(PainPoints)
        self.comment_verbatim_pipe = comment_verbatim_prompt | self.llm.with_structured_output(PainPoints)
        self.states: list[State] = self.build_states()

    def post_verbatim_extractor(self, state: State) -> dict:
        try:
            response = self.post_verbatim_pipe.invoke({
                "post_title": state["post_title"],
                "post_descr": state["post_descr"],
            })
            return {"pain_points": [
                PainPoint(post_id=state["post_id"], verbatim=pp.verbatim, pain_point_reformulated=pp.pain_point_reformulated, urgency=pp.urgency)
                for pp in response.pain_points
            ]}
        except Exception as e:
            if "rate_limit_error" in str(e):
                raise
            return {"pain_points": []}

    @staticmethod
    def get_all_comments(comments: Optional[list[Comment]]) -> list[Comment]:
        all_comments = []
        if comments:
            for comment in comments:
                all_comments.append(comment)
                if comment.get("sub_comments"):
                    all_comments.extend(
                        Workflow.get_all_comments(comment["sub_comments"])
                    )
        return all_comments

    @staticmethod
    def spawn_comment_workers(state: State) -> list[Send]:
        return [
            Send("comment_verbatim_extractor", {"post_id": state["post_id"], "post_title": state["post_title"], "post_descr": state["post_descr"], "comment": c})
            for c in Workflow.get_all_comments(state["comments"])
        ]

    def comment_verbatim_extractor(self, state: CommentWorkerState) -> dict:
        try:
            response = self.comment_verbatim_pipe.invoke({
                "post_title": state["post_title"],
                "post_descr": state["post_descr"],
                "comment": state["comment"]["text"],
            })
            return {"pain_points": [
                PainPoint(post_id=state["post_id"], verbatim=pp.verbatim, pain_point_reformulated=pp.pain_point_reformulated, urgency=pp.urgency)
                for pp in response.pain_points
            ]}
        except Exception as e:
            if "rate_limit_error" in str(e):
                raise
            return {"pain_points": []}

    @staticmethod
    def build_states() -> list[State]:
        def _map_comment(c: dict) -> Comment:
            return {"text": c["body"], "sub_comments": [_map_comment(r) for r in c.get("replies", [])]}

        return [
            {
                "post_id": p["id"],
                "post_title": p["title"], "post_descr": p.get("selftext", ""),
                "comments": [_map_comment(c) for c in p.get("comments", [])],
                "pain_points": [], }
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

        # Inner graph: one post → parallel comments + post extraction
        post_workflow = StateGraph(State)
        post_workflow.add_node("post_verbatim_extractor", self.post_verbatim_extractor, retry=retry_policy)
        post_workflow.add_node("comment_verbatim_extractor", self.comment_verbatim_extractor, retry=retry_policy)
        post_workflow.add_edge(START, "post_verbatim_extractor")
        post_workflow.add_conditional_edges(START, self.spawn_comment_workers)
        post_workflow.add_edge("post_verbatim_extractor", END)
        post_workflow.add_edge("comment_verbatim_extractor", END)
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
