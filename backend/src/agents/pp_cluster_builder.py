from __future__ import annotations

import operator
from typing import Annotated, TypedDict

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src.schemas.schema import MacroCluster, PainPoint

_EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"

strict_filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an extremely strict semantic deduplication engine. Your default answer is NO.

You receive a numbered list of pain points. Item 0 is the pivot.

Include an item ONLY if ALL three conditions are met:
1. Same root cause — the underlying technical or organisational failure is identical, not just similar.
2. Same problem scope — the frustration targets the same object, system, or process. A related but distinct system does not qualify.
3. Same type of failure — a usability issue and a reliability issue about the same system are NOT duplicates.

Reject an item if ANY of the following is true:
- It is merely thematically related or in the same domain.
- It shares keywords but describes a different failure mode.
- You have the slightest doubt.

You MUST always include item 0.
A cluster of 1 (only item 0) is perfectly valid and often the correct answer.""",
        ),
        (
            "human",
            "{candidates}",
        ),
    ]
)

macro_cluster_building_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a semantic clustering engine. You receive a list of micro-clusters, each identified by a pivot pain point.

Your task: group micro-clusters into macro-categories.

Rules:
- Only group clusters that address the SAME underlying problem — same root cause, same user frustration.
- Do NOT group clusters that are merely related, adjacent, or thematically close. When in doubt, keep them separate.
- Every micro-cluster must appear in exactly one macro-category (ungrouped clusters form a singleton category).""",
        ),
        (
            "human",
            "Micro-clusters:\n{micro_clusters}",
        ),
    ]
)


class FilterResult(BaseModel):
    matching_indices: list[int] = Field(
        description="Indices of candidates that describe exactly the same problem as the pivot"
    )


class MacroCategoryList(BaseModel):
    categories: list[list[int]] = Field(description="Each group is a list of micro-cluster indices")


class ClusterDescriptions(BaseModel):
    descriptions: list[str] = Field(description="One generalizing sentence per cluster, in order")


describe_clusters_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "For each numbered cluster, write one concise sentence that captures the shared underlying failure — not the topic, but the specific problem. Be precise.",
        ),
        ("human", "{clusters}"),
    ]
)

reflect_consolidation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a strict auditor. You receive a single proposed group of micro-clusters to merge.

Re-partition them into sub-groups where EVERY member shares the EXACT SAME underlying problem — same failure mode, same scope, same root cause.
A singleton (sub-group of 1) is valid and often correct. When in doubt, split.
Return every input index in exactly one sub-group.""",
        ),
        ("human", "{proposed_group}"),
    ]
)


class GroupVerification(BaseModel):
    subgroups: list[list[int]] = Field(
        description="Re-partition of the input indices into sub-groups; every index appears exactly once"
    )


class MicroCluster(TypedDict):
    members: list[PainPoint]
    description: str


class ClusteringState(TypedDict):
    pain_points: dict[int, PainPoint]
    micro_clusters: Annotated[list[MicroCluster], operator.add]
    described_clusters: list[MicroCluster]
    consolidation_input: list[MicroCluster]
    proposed_macro: list[list[int]]
    clusters: Annotated[list[MacroCluster], operator.add]


class ClusterBuilderWorkflow:
    def __init__(self, pain_points: list[PainPoint]):
        self.collection = self._build_collection(pain_points)
        self.llm = init_chat_model("claude-haiku-4-5")
        self.strict_filter_pipe = strict_filter_prompt | self.llm.with_structured_output(
            FilterResult
        )
        self.macro_cluster_building_pipe = (
            macro_cluster_building_prompt | self.llm.with_structured_output(MacroCategoryList)
        )
        self.describe_clusters_pipe = describe_clusters_prompt | self.llm.with_structured_output(
            ClusterDescriptions
        )
        self.reflect_consolidation_pipe = (
            reflect_consolidation_prompt | self.llm.with_structured_output(GroupVerification)
        )
        self.initial_state = ClusteringState(
            pain_points=dict(enumerate(pain_points)),
            micro_clusters=[],
            described_clusters=[],
            consolidation_input=[],
            proposed_macro=[],
            clusters=[],
        )
        self.built_graph = self._build_graph()

    @staticmethod
    def _build_collection(pain_points: list[PainPoint]):
        ef = SentenceTransformerEmbeddingFunction(model_name=_EMBED_MODEL, trust_remote_code=True)
        client = chromadb.EphemeralClient()
        collection = client.create_collection(
            "pp_clustering_collection",
            embedding_function=ef,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:search_ef": 2000,
                "hnsw:construction_ef": 400,
                "hnsw:M": 96,
            },
        )
        collection.add(
            ids=[str(i) for i in range(len(pain_points))],
            documents=[pp.pain_point_reformulated for pp in pain_points],
        )
        return collection

    def _remove_pain_points(self, state: ClusteringState, ids: list[int]) -> dict:
        self.collection.delete(ids=[str(i) for i in ids])
        return {"pain_points": {k: v for k, v in state["pain_points"].items() if k not in ids}}

    def knn_search(
        self, pivot: PainPoint, pain_points: dict[int, PainPoint], k: int = 15
    ) -> list[tuple[int, PainPoint]]:
        n = min(k, len(pain_points))
        if n == 0:
            return []
        results = self.collection.query(
            query_texts=[pivot.pain_point_reformulated],
            n_results=n,
        )
        return [(int(i), pain_points[int(i)]) for i in results["ids"][0] if int(i) in pain_points]

    def pick_and_filter(self, state: ClusteringState) -> dict:
        pp = state["pain_points"]
        results = self.knn_search(next(iter(pp.values())), pp)
        text = "\n".join(f"{j}. {p.pain_point_reformulated}" for j, (_, p) in enumerate(results))
        matched = self.strict_filter_pipe.invoke({"candidates": text}).matching_indices
        cluster_ids = [results[j][0] for j in matched if j < len(results)]
        removed = self._remove_pain_points(state, cluster_ids)
        if len(cluster_ids) <= 2:
            return removed
        cluster = MicroCluster(
            members=[pp[i] for i in cluster_ids], description=results[0][1].pain_point_reformulated
        )
        return removed | {"micro_clusters": [cluster]}

    def describe_clusters(self, state: ClusteringState) -> dict:
        clusters = state["micro_clusters"]
        text = "\n\n".join(
            f"Cluster {i}:\n" + "\n".join(f"- {pp.pain_point_reformulated}" for pp in c["members"])
            for i, c in enumerate(clusters)
        )
        descs = self.describe_clusters_pipe.invoke({"clusters": text}).descriptions
        return {
            "described_clusters": [
                MicroCluster(members=c["members"], description=descs[i])
                for i, c in enumerate(clusters)
                if i < len(descs)
            ]
        }

    def consolidate(self, state: ClusteringState) -> dict:
        top = sorted(state["described_clusters"], key=lambda c: len(c["members"]), reverse=True)[
            :15
        ]
        text = "\n".join(
            f"{i}. {c['description']} ({len(c['members'])} members)" for i, c in enumerate(top)
        )
        result = self.macro_cluster_building_pipe.invoke({"micro_clusters": text})
        return {"consolidation_input": top, "proposed_macro": result.categories}

    def reflect_consolidation(self, state: ClusteringState) -> dict:
        top = state["consolidation_input"]
        macro = []
        for ids in state["proposed_macro"]:
            valid = [i for i in ids if i < len(top)]
            if len(valid) <= 1:
                if valid:
                    macro.append(
                        MacroCluster(
                            description=top[valid[0]]["description"],
                            pain_points=[*top[valid[0]]["members"]],
                        )
                    )
                continue
            text = "\n".join(f"{j}. {top[i]['description']}" for j, i in enumerate(valid))
            for subgroup in self.reflect_consolidation_pipe.invoke(
                {"proposed_group": text}
            ).subgroups:
                real = [valid[j] for j in subgroup if j < len(valid)]
                if real:
                    macro.append(
                        MacroCluster(
                            description=top[real[0]]["description"],
                            pain_points=[pp for i in real for pp in top[i]["members"]],
                        )
                    )
        return {"clusters": sorted(macro, key=lambda c: len(c["pain_points"]), reverse=True)}

    def _should_continue(self, state: ClusteringState) -> str:
        return "loop" if state["pain_points"] else "done"

    def run(self) -> list[MacroCluster]:
        n = len(self.initial_state["pain_points"])
        config = {"recursion_limit": n + 10}
        return self.built_graph.invoke(self.initial_state, config=config)["clusters"]

    def _build_graph(self):
        graph = StateGraph(ClusteringState)
        graph.add_node("pick_and_filter", self.pick_and_filter)
        graph.add_node("describe_clusters", self.describe_clusters)
        graph.add_node("consolidate", self.consolidate)
        graph.add_node("reflect_consolidation", self.reflect_consolidation)
        graph.add_edge(START, "pick_and_filter")
        graph.add_conditional_edges(
            "pick_and_filter",
            self._should_continue,
            {"loop": "pick_and_filter", "done": "describe_clusters"},
        )
        graph.add_edge("describe_clusters", "consolidate")
        graph.add_edge("consolidate", "reflect_consolidation")
        graph.add_edge("reflect_consolidation", END)
        return graph.compile()
