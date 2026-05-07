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

MAX_REFLECTION_ITER = 2


THREADS_PATH = Path("data/processed/r_ciso_threads.jsonl")
OUTPUT_PATH = Path("data/processed/r_ciso_pain_points.jsonl")
MODEL = "claude-sonnet-4-6"

llm = init_chat_model(MODEL)


post_verbatim_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Tu reçois le titre et la description d'un post Reddit.
Découpe le contenu en un ou plusieurs verbatims. Chaque verbatim est une citation brute, mot pour mot, susceptible de contenir un pain point ou un besoin exprimé par l'auteur.
N'invente rien et ne reformule pas : extrais uniquement ce qui est présent dans le texte.""",
        ),
        ("human", "Titre: {post_title}\n\nDescription: {post_descr}{feedback}"),
    ]
)


comment_verbatim_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Tu es spécialisé en product management et tu cherches des besoins ou des douleurs de clients sur reddit.
Tu reçois (à titre de contexte uniquement) un post Reddit, puis un commentaire.
Découpe UNIQUEMENT le commentaire en un ou plusieurs verbatims. N'extrais AUCUN verbatim du post — il sert seulement de contexte.
Chaque verbatim est une citation brute, mot pour mot, susceptible de contenir un pain point ou un besoin.
Chaque verbatim retenu doit contenir une douleur ou un besoin exprimé implicitement ou explicitement 
N'invente rien et ne reformule pas.""",
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


class Extraction(BaseModel):
    pain_points: list[PainPoint]


class Verbatims(BaseModel):
    verbatims: list[str] = Field(
        description="Citations brutes potentiellement porteuses d'un pain point ou d'un besoin"
    )


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

    post_verbatims: Annotated[list[str], add]
    post_verbatim_feedback: str
    post_verbatim_iterations: int

    comment_verbatims: Annotated[list[str], add]
    comment_verbatim_feedback: str
    comment_verbatim_iterations: int

    extraction_feedback: str
    extractor_iterations: int

    pain_points: Annotated[list[PainPoint], add]
    reformulation_feedback: str


verbatim_structured_llm = llm.with_structured_output(Verbatims)
extractor_llm = llm.with_structured_output(Extraction)
revisor_llm = llm.with_structured_output(Extraction)
reflection_llm = llm.with_structured_output(ReflectionResult)


post_verbatim_pipe = post_verbatim_prompt | llm
comment_verbatim_pipe = comment_verbatim_prompt | verbatim_structured_llm
post_verbatim_reflection_pipe = post_verbatim_reflection_prompt | llm
comment_verbatim_reflection_pipe = comment_verbatim_reflection_prompt | llm
extraction_reflection_pipe = extraction_reflection_prompt | llm
reformulation_reflection_pipe = reformulation_reflection_prompt | llm


def post_verbatim_extractor(state: State):
    pass


def post_verbatim_reflector(state: State):
    pass


def comment_verbatim_reflector(state: State):
    pass


def extraction_reflector(state: State):
    pass


def reformulation_reflector(state: State):
    pass


def get_all_comments(comments: Optional[list[Comment]]) -> list[Comment]:
    """Recursively extracts all comments including sub-comments."""
    all_comments = []
    if comments is not None:
        for comment in comments:
            all_comments.append(comment)
            if comment.get("sub_comments"):
                all_comments.extend(get_all_comments(comment["sub_comments"]))
    return all_comments


def spawn_comment_workers(state: State) -> list[Send]:
    """Returns one Send per comment — creates N parallel branches."""
    all_comments = get_all_comments(state["comments"])
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
        for c in all_comments
    ]


def comment_verbatim_extractor(state: CommentWorkerState) -> dict:
    """Runs once per comment & sub_comments, all in parallel."""
    response = comment_verbatim_pipe.invoke(
        {
            "post_title": state["post_title"],
            "post_descr": state["post_descr"],
            "comment": state["comment"]["text"],
            "feedback": state["feedback"],
        }
    )
    return {"comment_verbatims": response.verbatims}


workflow = StateGraph(State)
workflow.add_node("comment_verbatim_extractor", comment_verbatim_extractor)
workflow.add_conditional_edges(START, spawn_comment_workers)
graph = workflow.compile()
