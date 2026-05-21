import json
import time
from pathlib import Path

import httpx

BASE_URL = "https://arctic-shift.photon-reddit.com"


def _paginate(client: httpx.Client, endpoint: str, params: dict) -> list[dict]:
    all_records: list[dict] = []
    current_after = params["after"]

    while True:
        resp = client.get(
            f"{BASE_URL}/{endpoint}", params={**params, "after": current_after}, timeout=30
        )
        resp.raise_for_status()

        remaining = resp.headers.get("X-RateLimit-Remaining")
        if remaining is not None and int(remaining) < 5:
            time.sleep(float(resp.headers.get("X-RateLimit-Reset", 10)))

        records = resp.json().get("data", [])
        if not records:
            break

        all_records.extend(records)

        if len(records) < 100:
            break

        current_after = str(int(records[-1]["created_utc"]) + 1)
        time.sleep(0.5)

    return all_records


def fetch_posts(subreddit: str, after: str, before: str) -> list[dict]:
    with httpx.Client() as client:
        return _paginate(
            client,
            "api/posts/search",
            {
                "subreddit": subreddit,
                "after": after,
                "before": before,
                "limit": "auto",
                "sort": "asc",
            },
        )


def fetch_comments(subreddit: str, after: str, before: str) -> list[dict]:
    with httpx.Client() as client:
        return _paginate(
            client,
            "api/comments/search",
            {
                "subreddit": subreddit,
                "after": after,
                "before": before,
                "limit": "auto",
                "sort": "asc",
            },
        )


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    out = Path(__file__).parent.parent.parent / "tests" / "data"
    posts = fetch_posts(subreddit="ciso", after="2026-04-01", before="2026-04-30")
    save_jsonl(posts, out / "r_ciso_2026-04-01_2026-04-30_posts.jsonl")
    comments = fetch_comments(subreddit="ciso", after="2026-04-01", before="2026-04-30")
    save_jsonl(comments, out / "r_ciso_2026-04-01_2026-04-30_comments.jsonl")
