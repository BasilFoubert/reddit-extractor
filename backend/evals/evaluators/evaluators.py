from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.core.llm import get_llm


class _Score(BaseModel):
    score: int = Field(description="1 if the pain point belongs to the cluster, 0 otherwise")


_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a strict cluster assignment judge. Score 1 if the pain point belongs to the cluster, 0 otherwise."),
    ("human", "Pain point: {pain_point_reformulated}\nCluster: {cluster_description}"),
])

_pipe = _prompt | get_llm().with_structured_output(_Score)


def cluster_assignment_judge(outputs: dict) -> dict:
    result = _pipe.invoke(outputs)
    return {"key": "cluster_assignment", "score": result.score}
