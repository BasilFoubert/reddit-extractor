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

llm = init_chat_model(
    MODEL,
)

evaluator_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """À partir de ce contenu uniquement, extrais chaque douleur
exprimée mot pour mot par les auteurs. Pour chaque douleur reformule de manière
claire la douleur ou le besoin exprimé, soit aussi spécifique que possible en
restant concis. Ne cherche pas d'informations extérieures."""
    ),          
    ("human", "{post_and_comments}")
])

revisor_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """Tu reçois un contenu source et une liste de pain points extraits.
Vérifie pour chaque pain point que le verbatim est bien présent ou clairement impliqué dans le contenu source, et que la reformulation y correspond fidèlement sans rien ajouter.
Supprime tout pain point inventé, vague ou mal reformulé.
Retourne uniquement les pain points valides sans les modifier."""
    ),
    ("human", "{post_and_comments}\n\n---\n{pain_points}")
])

post_verbatim_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """Tu reçois le titre et la description d'un post Reddit.
Découpe le contenu en un ou plusieurs verbatims. Chaque verbatim est une citation brute, mot pour mot, susceptible de contenir un pain point ou un besoin exprimé par l'auteur.
N'invente rien et ne reformule pas : extrais uniquement ce qui est présent dans le texte."""
    ),
    ("human", "Titre: {post_title}\n\nDescription: {post_descr}{feedback}")
])

comment_verbatim_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """Tu reçois (à titre de contexte uniquement) un post Reddit, puis un commentaire et ses sous-commentaires.
Découpe UNIQUEMENT le commentaire et ses sous-commentaires en un ou plusieurs verbatims. N'extrais AUCUN verbatim du post — il sert seulement de contexte.
Chaque verbatim est une citation brute, mot pour mot, susceptible de contenir un pain point ou un besoin.
N'invente rien et ne reformule pas."""
    ),
    ("human",
     "Contexte (post, ne pas extraire) :\nTitre: {post_title}\nDescription: {post_descr}\n\n---\n\nCommentaire :\n{comment}\n\nSous-commentaires :\n{sub_comments}{feedback}")
])

post_verbatim_reflection_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """Tu reçois le contenu d'un post Reddit (titre + description) et la liste des verbatims qui en ont été extraits.
Vérifie que :
- chaque verbatim provient bien du post (citation mot pour mot ou très proche),
- aucun passage du post porteur d'un pain point ou d'un besoin n'a été oublié,
- aucun passage neutre, descriptif ou hors-sujet n'a été retenu à tort.
Réponds approved=true uniquement si l'extraction est exhaustive et fidèle. Sinon approved=false avec un feedback précis indiquant ce qui doit être corrigé."""
    ),
    ("human", "Post:\nTitre: {post_title}\n\nDescription: {post_descr}\n\n---\n\nVerbatims extraits:\n{verbatims}")
])

comment_verbatim_reflection_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """Tu reçois (à titre de contexte) un post Reddit, puis un commentaire et ses sous-commentaires, ainsi que la liste des verbatims qui en ont été extraits.
Vérifie que :
- chaque verbatim provient bien du commentaire ou des sous-commentaires (PAS du post),
- chaque verbatim est une citation mot pour mot ou très proche,
- aucun passage porteur d'un pain point ou d'un besoin n'a été oublié,
- aucun passage neutre, descriptif ou hors-sujet n'a été retenu à tort.
Réponds approved=true uniquement si l'extraction est exhaustive et fidèle. Sinon approved=false avec un feedback précis indiquant ce qui doit être corrigé."""
    ),
    ("human",
     "Contexte (post, ne doit PAS être source des verbatims) :\nTitre: {post_title}\nDescription: {post_descr}\n\n---\n\nCommentaire :\n{comment}\n\nSous-commentaires :\n{sub_comments}\n\n---\n\nVerbatims extraits:\n{verbatims}")
])

extraction_reflection_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """Tu reçois la liste des verbatims candidats (extraits du post et des commentaires) et la liste des pain points qui en ont été extraits.
Vérifie que :
- chaque pain point correspond bien à un verbatim présent dans la source (pas d'invention),
- aucun verbatim porteur d'un pain point ou d'un besoin n'a été oublié,
- aucun verbatim sans pain point ni besoin n'a été retenu à tort.
Réponds approved=true uniquement si l'extraction est exhaustive et fidèle. Sinon approved=false avec un feedback précis indiquant ce qui doit être corrigé."""
    ),
    ("human", "Verbatims:\n{verbatims}\n\n---\n\nPain points extraits:\n{pain_points}")
])

reformulation_reflection_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """Tu reçois une liste de pain points (verbatim + reformulation).
Vérifie que chaque reformulation est :
- claire,
- spécifique,
- concise,
- fidèle au verbatim sans rien y ajouter.
Réponds approved=true uniquement si toutes les reformulations sont correctes. Sinon approved=false avec un feedback précis indiquant lesquelles doivent être corrigées et comment."""
    ),
    ("human", "Pain points:\n{pain_points}")
])

class PainPoint(BaseModel):
    verbatim: str = Field(description="Citation exacte de la douleur dans le texte source")
    pain_point_reformulated: str = Field(description="Reformulation claire et concise de la douleur")

class Extraction(BaseModel):
    pain_points: list[PainPoint]

class Verbatims(BaseModel):
    verbatims: list[str] = Field(description="Citations brutes potentiellement porteuses d'un pain point ou d'un besoin")

class ReflectionResult(BaseModel):
    approved: bool = Field(description="True si l'étape est correctement effectuée")
    feedback: str = Field(description="Si non approuvé, retours précis pour corriger. Vide si approuvé.")

class Comment(TypedDict):
    text: str
    sub_comments: Optional[list[Comment]]
class State(TypedDict):
    post_title: str
    post_descr: str
    comments: list[Comment]
    post_verbatims: Annotated[list[str], add]
    post_verbatim_feedback: str
    comment_verbatims: Annotated[list[str], add]
    comment_verbatim_feedback: str
    pain_points: Annotated[list[PainPoint], add]
    extraction_feedback: str
    reformulation_feedback: str
    post_verbatim_iterations: int
    comment_verbatim_iterations: int
    extractor_iterations: int


class StateUpdate(TypedDict, total=False):
    post_verbatims: list[str]
    comment_verbatims: list[str]
    pain_points: list[PainPoint]
    post_verbatim_feedback: str
    comment_verbatim_feedback: str
    extraction_feedback: str
    reformulation_feedback: str
    post_verbatim_iterations: int
    comment_verbatim_iterations: int
    extractor_iterations: int

verbatim_llm = llm.with_structured_output(Verbatims)
extractor_llm = llm.with_structured_output(Extraction)
revisor_llm = llm.with_structured_output(Extraction)
reflection_llm = llm.with_structured_output(ReflectionResult)


def _flatten_sub_comments(subs: list[Comment] | None, depth: int = 0) -> str:
    if not subs:
        return ""
    lines: list[str] = []
    for s in subs:
        lines.append("  " * depth + f"- {s['text']}")
        if s.get("sub_comments"):
            lines.append(_flatten_sub_comments(s["sub_comments"], depth + 1))
    return "\n".join(lines)


def _all_verbatims(state: State) -> str:
    verbatims: list[str] = (state.get("post_verbatims") or []) + (state.get("comment_verbatims") or [])
    return "\n".join(f"- {v}" for v in verbatims) or "(aucun)"


def _pain_points_payload(state: State) -> str:
    pain_points: list[PainPoint] = state.get("pain_points") or []
    return json.dumps(
        [p.model_dump() for p in pain_points],
        ensure_ascii=False,
        indent=2,
    )


def _feedback_block(feedback: str) -> str:
    return f"\n\n---\n\nRetours à corriger lors de cette nouvelle tentative :\n{feedback}" if feedback else ""


def post_verbatim_extractor(state: State) -> StateUpdate:
    feedback = state.get("post_verbatim_feedback") or ""
    result = Verbatims.model_validate((post_verbatim_prompt | verbatim_llm).invoke({
        "post_title": state["post_title"],
        "post_descr": state["post_descr"],
        "feedback": _feedback_block(feedback),
    }))
    return {
        "post_verbatims": result.verbatims,
        "post_verbatim_iterations": state.get("post_verbatim_iterations", 0) + 1,
        "post_verbatim_feedback": "",
    }


def post_verbatim_reflector(state: State) -> StateUpdate:
    verbatims_text = "\n".join(f"- {v}" for v in (state.get("post_verbatims") or [])) or "(aucun)"
    result = ReflectionResult.model_validate((post_verbatim_reflection_prompt | reflection_llm).invoke({
        "post_title": state["post_title"],
        "post_descr": state["post_descr"],
        "verbatims": verbatims_text,
    }))
    return {"post_verbatim_feedback": "" if result.approved else result.feedback}


def comment_verbatim_extractor(state: State) -> StateUpdate:
    sub_text = _flatten_sub_comments(state.get("sub_comments")) or "(aucun)"
    feedback = state.get("comment_verbatim_feedback") or ""
    result = Verbatims.model_validate((comment_verbatim_prompt | verbatim_llm).invoke({
        "post_title": state["post_title"],
        "post_descr": state["post_descr"],
        "comment": state["comment"],
        "sub_comments": sub_text,
        "feedback": _feedback_block(feedback),
    }))
    return {
        "comment_verbatims": result.verbatims,
        "comment_verbatim_iterations": state.get("comment_verbatim_iterations", 0) + 1,
        "comment_verbatim_feedback": "",
    }


def comment_verbatim_reflector(state: State) -> StateUpdate:
    sub_text = _flatten_sub_comments(state.get("sub_comments")) or "(aucun)"
    verbatims_text = "\n".join(f"- {v}" for v in (state.get("comment_verbatims") or [])) or "(aucun)"
    result = ReflectionResult.model_validate((comment_verbatim_reflection_prompt | reflection_llm).invoke({
        "post_title": state["post_title"],
        "post_descr": state["post_descr"],
        "comment": state["comment"],
        "sub_comments": sub_text,
        "verbatims": verbatims_text,
    }))
    return {"comment_verbatim_feedback": "" if result.approved else result.feedback}


def extractor(state: State) -> StateUpdate:
    feedbacks: list[str] = []
    if state.get("extraction_feedback"):
        feedbacks.append(f"[Extraction] {state['extraction_feedback']}")
    if state.get("reformulation_feedback"):
        feedbacks.append(f"[Reformulation] {state['reformulation_feedback']}")
    feedback_block = (
        "\n\nRetours à corriger lors de cette nouvelle tentative :\n" + "\n".join(feedbacks)
        if feedbacks else ""
    )
    context = f"Verbatims candidats :\n{_all_verbatims(state)}{feedback_block}"
    result = Extraction.model_validate((evaluator_prompt | extractor_llm).invoke({"post_and_comments": context}))
    return {
        "pain_points": result.pain_points,
        "extractor_iterations": state.get("extractor_iterations", 0) + 1,
        "extraction_feedback": "",
        "reformulation_feedback": "",
    }


def extraction_reflector(state: State) -> StateUpdate:
    result = ReflectionResult.model_validate((extraction_reflection_prompt | reflection_llm).invoke({
        "verbatims": _all_verbatims(state),
        "pain_points": _pain_points_payload(state),
    }))
    return {"extraction_feedback": "" if result.approved else result.feedback}


def reformulation_reflector(state: State) -> StateUpdate:
    result = ReflectionResult.model_validate((reformulation_reflection_prompt | reflection_llm).invoke({
        "pain_points": _pain_points_payload(state),
    }))
    return {"reformulation_feedback": "" if result.approved else result.feedback}


def revisor(state: State) -> StateUpdate:
    context = f"Titre: {state['post_title']}\n\nDescription: {state['post_descr']}\n\nCommentaire: {state['comment']}"
    payload = _pain_points_payload(state)
    result = Extraction.model_validate((revisor_prompt | revisor_llm).invoke({
        "post_and_comments": context,
        "pain_points": payload,
    }))
    return {"pain_points": result.pain_points}


def after_post_verbatim_reflection(state: State) -> str:
    if state.get("post_verbatim_feedback") and state.get("post_verbatim_iterations", 0) < MAX_REFLECTION_ITER:
        return "post_verbatim_extractor"
    return "comment_verbatim_extractor"


def after_comment_verbatim_reflection(state: State) -> str:
    if state.get("comment_verbatim_feedback") and state.get("comment_verbatim_iterations", 0) < MAX_REFLECTION_ITER:
        return "comment_verbatim_extractor"
    return "extractor"


def after_extraction_reflection(state: State) -> str:
    if not state.get("pain_points"):
        return END
    if state.get("extraction_feedback") and state.get("extractor_iterations", 0) < MAX_REFLECTION_ITER:
        return "extractor"
    return "reformulation_reflector"


def after_reformulation_reflection(state: State) -> str:
    if state.get("reformulation_feedback") and state.get("extractor_iterations", 0) < MAX_REFLECTION_ITER:
        return "extractor"
    return "revisor"


workflow = StateGraph(State)

# Nodes
workflow.add_node("post_verbatim_extractor", post_verbatim_extractor)
workflow.add_node("post_verbatim_reflector", post_verbatim_reflector)
workflow.add_node("comment_verbatim_extractor", comment_verbatim_extractor)
workflow.add_node("comment_verbatim_reflector", comment_verbatim_reflector)
workflow.add_node("extractor", extractor)
workflow.add_node("extraction_reflector", extraction_reflector)
workflow.add_node("reformulation_reflector", reformulation_reflector)
workflow.add_node("revisor", revisor)

# Edges
workflow.add_edge(START, "post_verbatim_extractor")
workflow.add_edge("post_verbatim_extractor", "post_verbatim_reflector")
workflow.add_conditional_edges(
    "post_verbatim_reflector",
    after_post_verbatim_reflection,
    {"post_verbatim_extractor": "post_verbatim_extractor", "comment_verbatim_extractor": "comment_verbatim_extractor"},
)
workflow.add_edge("comment_verbatim_extractor", "comment_verbatim_reflector")
workflow.add_conditional_edges(
    "comment_verbatim_reflector",
    after_comment_verbatim_reflection,
    {"comment_verbatim_extractor": "comment_verbatim_extractor", "extractor": "extractor"},
)
workflow.add_edge("extractor", "extraction_reflector")
workflow.add_conditional_edges(
    "extraction_reflector",
    after_extraction_reflection,
    {"extractor": "extractor", "reformulation_reflector": "reformulation_reflector", END: END},
)
workflow.add_conditional_edges(
    "reformulation_reflector",
    after_reformulation_reflection,
    {"extractor": "extractor", "revisor": "revisor"},
)
workflow.add_edge("revisor", END)

# Compile the graph
graph = workflow.compile()

