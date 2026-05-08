from __future__ import annotations
import json
from operator import add
from pathlib import Path
from typing import TypedDict, Optional, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.utils import load_jsonl


THREADS_PATH = Path("data/processed/r_ciso_threads.jsonl")
OUTPUT_PATH = Path("data/processed/r_ciso_pain_points.jsonl")


post_verbatim_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You receive the title and description of a Reddit post.
Identify every pain or need expressed by the author, implicitly or explicitly.
For each one:
- verbatim: exact word-for-word quote from the text
- reformulation: express the pain or need clearly and concisely in English, with enough context to be understood without reading the post. Minimum words, maximum clarity.
Do not invent anything: the verbatim must exist as-is in the text.""",
        ),
        ("human", "Titre: {post_title}\n\nDescription: {post_descr}{feedback}"),
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
- reformulation: express the pain or need clearly and concisely in English, using both the post and comment context to be understood without reading them. Minimum words, maximum clarity.
Do not invent anything: the verbatim must exist as-is in the comment.""",
        ),
        (
            "human",
            "Contexte (post, ne pas extraire) :\nTitre: {post_title}\nDescription: {post_descr}\n\n---\n\nCommentaire :\n{comment}\n\n{feedback}",
        ),
    ]
)


post_verbatim_reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Tu reçois le contenu d'un post Reddit (titre + description) et la liste des verbatims qui en ont été extraits.
Vérifie que :
- chaque verbatim provient bien du post (citation mot pour mot ou très proche),
- aucun passage du post porteur d'un pain point ou d'un besoin n'a été oublié,
- aucun passage neutre, descriptif ou hors-sujet n'a été retenu à tort.
Réponds approved=true uniquement si l'extraction est exhaustive et fidèle. Sinon approved=false avec un feedback précis indiquant ce qui doit être corrigé.""",
        ),
        (
            "human",
            "Post:\nTitre: {post_title}\n\nDescription: {post_descr}\n\n---\n\nVerbatims extraits:\n{verbatims}",
        ),
    ]
)


comment_verbatim_reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Tu reçois (à titre de contexte) un post Reddit, puis un commentaire et ses sous-commentaires, ainsi que la liste des verbatims qui en ont été extraits.
Vérifie que :
- chaque verbatim provient bien du commentaire ou des sous-commentaires (PAS du post),
- chaque verbatim est exactement une citation mot pour mot,
- aucun passage porteur d'un pain point ou d'un besoin n'a été oublié,
- aucun passage neutre, descriptif ou hors-sujet n'a été retenu à tort.
Réponds approved=true uniquement si l'extraction est exhaustive et fidèle. Sinon approved=false avec un feedback précis indiquant ce qui doit être corrigé.""",
        ),
        (
            "human",
            "Contexte (post, ne doit PAS être source des verbatims) :\nTitre: {post_title}\nDescription: {post_descr}\n\n---\n\nCommentaire :\n{comment}\n\nSous-commentaires :\n{sub_comments}\n\n---\n\nVerbatims extraits:\n{verbatims}",
        ),
    ]
)


extraction_reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Tu reçois la liste des verbatims candidats (extraits du post et des commentaires) et la liste des pain points qui en ont été extraits.
Vérifie que :
- chaque pain point correspond bien à un verbatim présent dans la source (pas d'invention),
- aucun verbatim porteur d'un pain point ou d'un besoin n'a été oublié,
- aucun verbatim sans pain point ni besoin n'a été retenu à tort.
Réponds approved=true uniquement si l'extraction est exhaustive et fidèle. Sinon approved=false avec un feedback précis indiquant ce qui doit être corrigé.""",
        ),
        (
            "human",
            "Verbatims:\n{verbatims}\n\n---\n\nPain points extraits:\n{pain_points}",
        ),
    ]
)

reformulation_reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Tu reçois une liste de pain points (verbatim + reformulation).
Vérifie que chaque reformulation est :
- claire,
- spécifique,
- concise,
- fidèle au verbatim sans rien y ajouter.
Réponds approved=true uniquement si toutes les reformulations sont correctes. Sinon approved=false avec un feedback précis indiquant lesquelles doivent être corrigées et comment.""",
        ),
        ("human", "Pain points:\n{pain_points}"),
    ]
)


class PainPoint(BaseModel):
    verbatim: str = Field(
        description="Citation exacte de la douleur dans le texte source"
    )
    pain_point_reformulated: str = Field(
        description="Reformulation claire et concise de la douleur"
    )


class PainPoints(BaseModel):
    pain_points: list[PainPoint]



class ReflectionResult(BaseModel):
    approved: bool = Field(description="True si l'étape est correctement effectuée")
    feedback: str = Field(
        description="Si non approuvé, retours précis pour corriger. Vide si approuvé."
    )


class Comment(TypedDict):
    text: str
    sub_comments: Optional[list[Comment]]


class CommentWorkerState(TypedDict):
    post_title: str
    post_descr: str
    comment: Comment
    feedback: str


class State(TypedDict):
    post_title: str
    post_descr: str
    comments: list[Comment]
    pain_points: Annotated[list[PainPoint], add]
    extraction_feedback: str
    extractor_iterations: int
    reformulation_feedback: str


class States(TypedDict):
    states_list: list[State]
    pain_points: Annotated[list[PainPoint], add]


# verbatim_structured_llm = llm.with_structured_output(Verbatims)
# extractor_llm = llm.with_structured_output(PainPoints)
# revisor_llm = llm.with_structured_output(PainPoints)
# reflection_llm = llm.with_structured_output(ReflectionResult)
#
#
# post_verbatim_pipe = post_verbatim_prompt | llm
# post_verbatim_reflection_pipe = post_verbatim_reflection_prompt | llm
# comment_verbatim_reflection_pipe = comment_verbatim_reflection_prompt | llm
# extraction_reflection_pipe = extraction_reflection_prompt | llm
# reformulation_reflection_pipe = reformulation_reflection_prompt | llm
#


class Workflow:
    MODEL = "claude-haiku-4-5"
    MAX_REFLECTION_ITER = 2

    def __init__(self):
        self.llm = init_chat_model(self.MODEL)
        self.post_verbatim_pipe = post_verbatim_prompt | self.llm.with_structured_output(PainPoints)
        self.comment_verbatim_pipe = comment_verbatim_prompt | self.llm.with_structured_output(PainPoints)
        self.states: list[State] = self.build_states()

    def post_verbatim_extractor(self, state: State) -> dict:
        response = self.post_verbatim_pipe.invoke({
            "post_title": state["post_title"],
            "post_descr": state["post_descr"],
            "feedback": state.get("extraction_feedback", ""),
        })
        return {"pain_points": response.pain_points}

    def post_verbatim_reflector(self, state: State):
        pass

    def comment_verbatim_reflector(self, state: State):
        pass

    def extraction_reflector(self, state: State):
        pass

    def reformulation_reflector(self, state: State):
        pass

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
            Send(
                "comment_verbatim_extractor",
                {
                    "post_title": state["post_title"],
                    "post_descr": state["post_descr"],
                    "comment": c,
                    "feedback": "",
                },
            )
            for c in Workflow.get_all_comments(state["comments"])
        ]

    def comment_verbatim_extractor(self, state: CommentWorkerState) -> dict:
        response = self.comment_verbatim_pipe.invoke(
            {
                "post_title": state["post_title"],
                "post_descr": state["post_descr"],
                "comment": state["comment"]["text"],
                "feedback": state["feedback"],
            }
        )
        return {"pain_points": response.pain_points}

    @staticmethod
    def build_states() -> list[State]:
        def _map_comment(c: dict) -> Comment:
            return {"text": c["body"], "sub_comments": [_map_comment(r) for r in c.get("replies", [])]}

        return [
            {
                "post_title": p["title"], "post_descr": p.get("selftext", ""),
                "comments": [_map_comment(c) for c in p.get("comments", [])],
                "extraction_feedback": "", "extractor_iterations": 0,
                "pain_points": [], "reformulation_feedback": "",
            }
            for p in load_jsonl(THREADS_PATH)
        ]

    @staticmethod
    def spawn_post_workers(state: States) -> list[Send]:
        return [Send("process_post", s) for s in state["states_list"]]

    def build_graph(self):
        # Inner graph: one post → parallel comments + post extraction
        post_workflow = StateGraph(State)
        post_workflow.add_node("post_verbatim_extractor", self.post_verbatim_extractor)
        post_workflow.add_node("comment_verbatim_extractor", self.comment_verbatim_extractor)
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
        self.built_graph= outer_workflow.compile()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    THREADS_PATH = Path("tests/data/small_subreddit.jsonl")
    THREADS_PATH_TEST = Path("tests/data/small_subreddit_verbatims.jsonl")
    wf = Workflow()
    wf.build_graph()
    result = wf.built_graph.invoke({"states_list": wf.states, "pain_points": []})
    print("=== PAIN POINTS ===")
    for pp in result["pain_points"]:
        print(f"  verbatim     : {pp.verbatim}")
        print(f"  reformulation: {pp.pain_point_reformulated}")
        print()
