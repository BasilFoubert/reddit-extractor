from __future__ import annotations

import pprint
import time
from typing import TypedDict

import httpx
from langsmith import traceable

from src.agents.pp_cluster_builder import ClusterBuilderWorkflow
from src.agents.pp_extractor import PPExtractorWorkflow
from src.schemas.schema import MacroCluster, PainPoint

_BASE_URL = "https://arctic-shift.photon-reddit.com"

_ENDPOINTS = {
    "posts": "api/posts/search",
    "comments": "api/comments/search",
}


_MIN_COMMENT_LENGTH = 50
_MIN_THREAD_SCORE = 3
_URGENCY_THRESHOLD = 6


_COMMENT_FIELDS = [
    "id",
    "author",
    "body",
    "score",
    "created_utc",
    "subreddit",
    "link_id",
    "parent_id",
    "permalink",
]

_POST_FIELDS = [
    "id",
    "author",
    "title",
    "selftext",
    "score",
    "upvote_ratio",
    "num_comments",
    "created_utc",
    "subreddit",
    "permalink",
]


class CommentNode(TypedDict):
    id: str
    author: str
    body: str
    score: int
    created_utc: int
    replies: list[CommentNode]


class Thread(TypedDict):
    id: str
    author: str
    title: str
    selftext: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: int
    subreddit: str
    permalink: str
    comments: list[CommentNode]


class Comment(TypedDict):
    id: str
    author: str
    body: str
    score: int
    created_utc: int
    subreddit: str
    link_id: str
    parent_id: str
    permalink: str


class Post(TypedDict):
    id: str
    author: str
    title: str
    selftext: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: int
    subreddit: str
    permalink: str


def _build_comment_node(raw: dict, replies: list[CommentNode]) -> CommentNode:
    return CommentNode(
        id=str(raw.get("id", "")),
        author=str(raw.get("author", "")),
        body=str(raw.get("body", "")),
        score=int(raw.get("score") or 0),
        created_utc=int(raw.get("created_utc") or 0),
        replies=replies,
    )


def _build_thread_list(posts: list[dict], comments: list[dict]) -> list[Thread]:
    by_parent: dict[str, list[dict]] = {}
    for c in comments:
        by_parent.setdefault(str(c.get("parent_id", "")), []).append(c)

    def build_replies(comment_id: str) -> list[CommentNode]:
        children = by_parent.get(f"t1_{comment_id}", [])
        return [
            _build_comment_node(child, build_replies(str(child["id"])))
            for child in sorted(children, key=lambda x: int(x.get("score") or 0), reverse=True)
        ]

    threads = []
    for post in posts:
        post_id = str(post["id"])
        top_comments = [
            _build_comment_node(c, build_replies(str(c["id"])))
            for c in sorted(
                by_parent.get(f"t3_{post_id}", []),
                key=lambda x: int(x.get("score") or 0),
                reverse=True,
            )
        ]
        threads.append(
            Thread(
                id=post_id,
                author=str(post.get("author", "")),
                title=str(post.get("title", "")),
                selftext=str(post.get("selftext", "")),
                score=int(post.get("score") or 0),
                upvote_ratio=float(post.get("upvote_ratio") or 0.0),
                num_comments=int(post.get("num_comments") or 0),
                created_utc=int(post.get("created_utc") or 0),
                subreddit=str(post.get("subreddit", "")),
                permalink=str(post.get("permalink", "")),
                comments=top_comments,
            )
        )

    return [t for t in threads if t["comments"] and t["score"] >= _MIN_THREAD_SCORE]


def _normalize_comment(r: dict) -> Comment:
    return Comment(
        id=str(r.get("id", "")),
        author=str(r.get("author", "")),
        body=str(r.get("body", "")).strip(),
        score=int(r.get("score") or 0),
        created_utc=int(r.get("created_utc") or 0),
        subreddit=str(r.get("subreddit", "")),
        link_id=str(r.get("link_id", "")),
        parent_id=str(r.get("parent_id", "")),
        permalink=str(r.get("permalink", "")),
    )


def _normalize_post(r: dict) -> Post:
    return Post(
        id=str(r.get("id", "")),
        author=str(r.get("author", "")),
        title=str(r.get("title", "")).strip(),
        selftext=str(r.get("selftext", "")).strip(),
        score=int(r.get("score") or 0),
        upvote_ratio=float(r.get("upvote_ratio") or 0.0),
        num_comments=int(r.get("num_comments") or 0),
        created_utc=int(r.get("created_utc") or 0),
        subreddit=str(r.get("subreddit", "")),
        permalink=str(r.get("permalink", "")),
    )


def _paginate(client: httpx.Client, endpoint: str, params: dict) -> list[dict]:
    all_records: list[dict] = []
    after = params["after"]

    while True:
        resp = client.get(f"{_BASE_URL}/{endpoint}", params={**params, "after": after}, timeout=30)
        resp.raise_for_status()

        remaining = resp.headers.get("X-RateLimit-Remaining")
        if remaining is not None and int(remaining) < 5:
            time.sleep(float(resp.headers.get("X-RateLimit-Reset", 10)))

        records = resp.json().get("data", [])
        all_records.extend(records)

        if len(records) < 100:
            break

        after = str(int(records[-1]["created_utc"]) + 1)
        time.sleep(0.5)

    return all_records


class ThreadsManagerService:
    def __init__(self):
        self.subreddit_name: str = ""
        self.start_date: str = ""
        self.end_date: str = ""
        self.raw_posts: list[dict] = []
        self.raw_comments: list[dict] = []
        self.posts: list[Post] = []
        self.comments: list[Comment] = []
        self.threads: list[Thread] = []
        self.pain_points: list[PainPoint] = []
        self.filtered_pp: list[PainPoint] = []
        self.clusters: list[MacroCluster] = []

    def __setstate__(self, state: dict) -> None:
        # backward compat: pickles saved before `clusters` was introduced
        self.__dict__.update(state)
        if not hasattr(self, "clusters"):
            self.clusters = []

    def set_subreddit_name(self, subreddit_name: str) -> None:
        self.subreddit_name = subreddit_name

    def set_start_date(self, start_date: str) -> None:
        self.start_date = start_date

    def set_end_date(self, end_date: str) -> None:
        self.end_date = end_date

    def download_subreddit(self, after: str, before: str) -> None:
        """Download subreddit posts and comments from Arctic Shift into memory."""
        params = {
            "subreddit": self.subreddit_name,
            "after": after,
            "before": before,
            "limit": "auto",
            "sort": "asc",
        }
        with httpx.Client() as client:
            self.raw_posts = _paginate(client, _ENDPOINTS["posts"], params)
            self.raw_comments = _paginate(client, _ENDPOINTS["comments"], params)

    def ingest_posts(self):
        """Normalize raw posts."""
        self.posts = [
            _normalize_post({k: r[k] for k in _POST_FIELDS if k in r}) for r in self.raw_posts
        ]

    def ingest_comments(self):
        """Filter short comments and normalize."""
        self.comments = [
            _normalize_comment({k: r[k] for k in _COMMENT_FIELDS if k in r})
            for r in self.raw_comments
            if len(str(r.get("body", ""))) >= _MIN_COMMENT_LENGTH
        ]

    def build_threads(self):
        """Assemble posts and comments into threads."""
        self.threads = _build_thread_list(self.posts, self.comments)

    @traceable(name="ThreadsManagerService.extract_pain_points", run_type="chain")
    def extract_pain_points(self):
        """Extract pain points from threads using LLM."""
        self.pain_points = PPExtractorWorkflow(threads=self.threads).run()

    def filter_pain_points(self, urgency_threshold: int = _URGENCY_THRESHOLD):
        """Filter pain points by urgency threshold."""
        self.filtered_pp = [pp for pp in self.pain_points if pp.urgency >= urgency_threshold]

    @traceable(name="ThreadsManagerService.spot_clusters", run_type="chain")
    def spot_clusters(self) -> None:
        source = self.filtered_pp if self.filtered_pp else self.pain_points
        self.clusters = ClusterBuilderWorkflow(pain_points=source).run()

    def flatten_dataset(self) -> list[dict]:
        return [
            {
                "verbatim": pp.verbatim,
                "pain_point_reformulated": pp.pain_point_reformulated,
                "cluster_description": c["description"],
            }
            for c in self.clusters
            for pp in c["pain_points"]
        ]

    def stats(self) -> None:
        total_clusters = len(self.clusters)
        pps_per_cluster = [len(c["pain_points"]) for c in self.clusters]
        pps_in_clusters = sum(pps_per_cluster)
        total_comments = sum(t["num_comments"] for t in self.threads)

        print(f"Subreddit                   : {self.subreddit_name}")
        print(f"Period                      : {self.start_date} → {self.end_date}")
        print()
        print(f"Threads                     : {len(self.threads)}")
        print(f"Comments                    : {total_comments}")
        print()
        print(f"Pain points                 : {len(self.pain_points)}")
        if total_clusters:
            print(f"Clusters                    : {total_clusters}")
            print(f"PP in clusters              : {pps_in_clusters}")
            print(f"Avg per cluster             : {pps_in_clusters / total_clusters:.1f}")
            print(f"Min & Max PP per cluster    : {min(pps_per_cluster)} / {max(pps_per_cluster)}")

    def short_print(self) -> None:
        """Print the first 10 items of each list attribute."""
        for name, items in [
            ("raw_posts", self.raw_posts),
            ("raw_comments", self.raw_comments),
            ("posts", self.posts),
            ("comments", self.comments),
            ("threads", self.threads),
            ("pain_points", self.pain_points),
            ("filtered_pp", self.filtered_pp),
        ]:
            print(f"\n--- {name} ({len(items)}) ---")
            for item in items[:10]:
                data = item.model_dump() if isinstance(item, PainPoint) else item
                pprint.pprint(data, sort_dicts=False)

    @traceable(name="ThreadsManagerService.run_pipeline", run_type="chain")
    def run_pipeline(self) -> None:
        """Run all pipeline steps in order."""
        self.download_subreddit(self.start_date, self.end_date)
        self.ingest_posts()
        self.ingest_comments()
        self.build_threads()
        self.extract_pain_points()
        self.filter_pain_points()
        self.spot_clusters()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # run full pipeline
    # service = ThreadsManagerService()
    # service.set_subreddit_name("ciso")
    # service.set_start_date(str(int(datetime(2025, 4, 1).timestamp())))
    # service.set_end_date(str(int(datetime(2025, 4, 30, 23, 59, 59).timestamp())))
    # service.run_pipeline()
    # service.short_print()

    # run only pain points clusters identification
    # import pickle
    # from pathlib import Path
    # _pkl_path = Path(__file__).parents[2] / "tests" / "data" / "ciso_2026-04-01_2026-04-30.pkl"
    # with open(_pkl_path, "rb") as f:
    #     service: ThreadsManagerService = pickle.load(f)

    # print(f"Loaded: {len(service.pain_points)} pain points, {len(service.filtered_pp)} filtered")
    # service.spot_clusters()
    # pprint.pprint([c for c in service.clusters], sort_dicts=False)

    # _out_path = _pkl_path.with_stem(_pkl_path.stem + "_clustered")
    # with open(_out_path, "wb") as f:
    #     pickle.dump(service, f)
    # print(f"Saved to: {_out_path}")
