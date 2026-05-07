import json
from pathlib import Path
from typing import TypedDict

# --- Config ---
MIN_BODY_LENGTH = 50

FIELDS_TO_KEEP = [
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

# Types
RawRecord = dict[str, str | int | float | bool | None]


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


# --- Functions ---


def load_jsonl(path: str | Path) -> list[RawRecord]:
    records: list[RawRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def filter_short_comments(
    records: list[RawRecord], min_length: int = MIN_BODY_LENGTH
) -> list[RawRecord]:
    return [r for r in records if len(str(r.get("body", ""))) >= min_length]


def filter_fields(
    records: list[RawRecord], fields: list[str] = FIELDS_TO_KEEP
) -> list[RawRecord]:
    return [{k: r[k] for k in fields if k in r} for r in records]


def normalize(record: RawRecord) -> Comment:
    return Comment(
        id=str(record.get("id", "")),
        author=str(record.get("author", "")),
        body=str(record.get("body", "")).strip(),
        score=int(record.get("score") or 0),
        created_utc=int(record.get("created_utc") or 0),
        subreddit=str(record.get("subreddit", "")),
        link_id=str(record.get("link_id", "")),
        parent_id=str(record.get("parent_id", "")),
        permalink=str(record.get("permalink", "")),
    )


def write_jsonl(records: list[Comment], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def ingest_comments(input_path: str | Path, output_path: str | Path) -> int:
    records = load_jsonl(input_path)
    records = filter_short_comments(records)
    records = filter_fields(records)
    comments = [normalize(r) for r in records]
    write_jsonl(comments, output_path)
    return len(comments)


if __name__ == "__main__":
    n = ingest_comments(
        "data/raw/r_ciso_comments.jsonl", "data/processed/r_ciso_comments.jsonl"
    )
    print(f"{n} comments written")
