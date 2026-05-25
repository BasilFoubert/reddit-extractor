from __future__ import annotations

import time
from typing import TypedDict

import httpx

from src.agents.extract_pain_points import Workflow
from src.schemas.schema import PainPoint

_BASE_URL = "https://arctic-shift.photon-reddit.com"

_ENDPOINTS = {
    "posts": "api/posts/search",
    "comments": "api/comments/search",
}


_MIN_COMMENT_LENGTH = 50
_MIN_THREAD_SCORE = 3

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
    def __init__(self, subreddit_name: str):
        self.subreddit_name = subreddit_name
        self.raw_posts: list[dict] = []
        self.raw_comments: list[dict] = []
        self.posts: list[Post] = []
        self.comments: list[Comment] = []
        self.threads: list[Thread] = []
        self.pain_points: list[PainPoint] = []

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

    def ingest_posts(self) -> int:
        """Normalize raw posts. Returns the number of posts."""
        self.posts = [
            _normalize_post({k: r[k] for k in _POST_FIELDS if k in r}) for r in self.raw_posts
        ]
        return len(self.posts)

    def ingest_comments(self) -> int:
        """Filter short comments, normalize. Returns the number of comments."""
        self.comments = [
            _normalize_comment({k: r[k] for k in _COMMENT_FIELDS if k in r})
            for r in self.raw_comments
            if len(str(r.get("body", ""))) >= _MIN_COMMENT_LENGTH
        ]
        return len(self.comments)

    def build_threads(self) -> int:
        """Assemble posts and comments into threads. Returns the number of threads."""
        self.threads = _build_thread_list(self.posts, self.comments)
        return len(self.threads)

    def extract_pain_points(self) -> int:
        """Extract pain points from threads using LLM. Returns the number of pain points."""
        self.pain_points = Workflow(threads=self.threads).run()
        return len(self.pain_points)

    def filter_pain_points(self) -> int:
        """Filter pain points by urgency threshold. Returns the number of pain points kept."""
        pass

    def run_pipeline(self) -> None:
        """Run all pipeline steps in order."""
        pass
