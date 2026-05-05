from __future__ import annotations
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TypedDict

POSTS_PATH = Path("data/processed/r_ciso_posts.jsonl")
COMMENTS_PATH = Path("data/processed/r_ciso_comments.jsonl")
OUTPUT_PATH = Path("data/processed/r_ciso_threads.jsonl")

RawRecord = dict[str, str | int | float | bool | None]

@dataclass
class Comment:
    id: str
    author: str
    body: str
    score: int
    created_utc: int
    replies: list[Comment] = field(default_factory=list) # pyright: ignore[reportUnknownVariableType]


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
    comments: list[Comment]


def load_jsonl(path: Path) -> list[RawRecord]:
    records: list[RawRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_comment(raw: RawRecord, replies: list[Comment]) -> Comment:
    return Comment(
        id=str(raw.get("id", "")),
        author=str(raw.get("author", "")),
        body=str(raw.get("body", "")),
        score=int(raw.get("score") or 0),
        created_utc=int(raw.get("created_utc") or 0),
        replies=replies,
    )


def build_threads(posts: list[RawRecord], comments: list[RawRecord]) -> list[Thread]:
    by_parent: dict[str, list[RawRecord]] = {}
    for c in comments:
        parent = str(c.get("parent_id", ""))
        by_parent.setdefault(parent, []).append(c)

    def build_replies(comment_id: str) -> list[Comment]:
        children = by_parent.get(f"t1_{comment_id}", [])
        return [
            build_comment(child, build_replies(str(child["id"])))
            for child in sorted(children, key=lambda x: int(x.get("score") or 0), reverse=True)
        ]

    threads: list[Thread] = []
    for post in posts:
        post_id = str(post["id"])
        top_level = by_parent.get(f"t3_{post_id}", [])
        top_comments = [
            build_comment(c, build_replies(str(c["id"])))
            for c in sorted(top_level, key=lambda x: int(x.get("score") or 0), reverse=True)
        ]
        threads.append(Thread(
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
        ))

    return threads


def write_jsonl(threads: list[Thread], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for thread in threads:
            serializable = dict(thread)
            serializable["comments"] = [asdict(c) for c in thread["comments"]]
            f.write(json.dumps(serializable, ensure_ascii=False) + "\n")


def build_thread_file(
    posts_path: Path = POSTS_PATH,
    comments_path: Path = COMMENTS_PATH,
    output_path: Path = OUTPUT_PATH,
) -> int:
    posts = load_jsonl(posts_path)
    comments = load_jsonl(comments_path)
    threads = build_threads(posts, comments)
    write_jsonl(threads, output_path)
    return len(threads)


if __name__ == "__main__":
    n = build_thread_file()
    print(f"{n} threads written to {OUTPUT_PATH}")
