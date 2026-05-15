import json
from pathlib import Path
from typing import TypedDict

# --- Config ---
FIELDS_TO_KEEP = [
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

# Types
RawRecord = dict[str, str | int | float | bool | None]


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


# --- Functions ---


def load_jsonl(path: str | Path) -> list[RawRecord]:
    records: list[RawRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def filter_fields(records: list[RawRecord], fields: list[str] = FIELDS_TO_KEEP) -> list[RawRecord]:
    return [{k: r[k] for k in fields if k in r} for r in records]


def normalize(record: RawRecord) -> Post:
    return Post(
        id=str(record.get("id", "")),
        author=str(record.get("author", "")),
        title=str(record.get("title", "")).strip(),
        selftext=str(record.get("selftext", "")).strip(),
        score=int(record.get("score") or 0),
        upvote_ratio=float(record.get("upvote_ratio") or 0.0),
        num_comments=int(record.get("num_comments") or 0),
        created_utc=int(record.get("created_utc") or 0),
        subreddit=str(record.get("subreddit", "")),
        permalink=str(record.get("permalink", "")),
    )


def write_jsonl(records: list[Post], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def ingest_posts(input_path: str | Path, output_path: str | Path) -> int:
    records = load_jsonl(input_path)
    records = filter_fields(records)
    posts = [normalize(r) for r in records]
    write_jsonl(posts, output_path)
    return len(posts)


if __name__ == "__main__":
    n = ingest_posts("data/raw/r_ciso_posts.jsonl", "data/processed/r_ciso_posts.jsonl")
    print(f"{n} posts written")
