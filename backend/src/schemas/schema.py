from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel, Field


class PainPoint(BaseModel):
    post_id: str = ""
    verbatim: str = Field(description="Exact quote of the pain point from the source text")
    pain_point_reformulated: str = Field(
        description="Precise, self-contained reformulation including who, what, why, and context"
    )
    urgency: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Urgency level from 1 (vague or hypothetical need) to 10 (actively seeking an immediate solution, critical pain)",
    )


class PainSummary(BaseModel):
    index: int = Field(description="Unique index identifying this pain point")
    description: str = Field(description="Brief description of the pain, 10 words max")


class PostPainSummary(BaseModel):
    pain_summaries: list[PainSummary]


class Comment(TypedDict):
    text: str
    sub_comments: list[Comment] | None


class MacroCluster(TypedDict):
    label: str
    description: str
    pain_points: list[PainPoint]
