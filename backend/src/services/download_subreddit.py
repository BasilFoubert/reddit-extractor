import time
from pathlib import Path

import httpx

from src.core.utils import save_jsonl

BASE_URL = "https://arctic-shift.photon-reddit.com"

ENDPOINTS = {
    "posts": "api/posts/search",
    "comments": "api/comments/search",
}


def _paginate(client: httpx.Client, endpoint: str, params: dict) -> list[dict]:
    all_records: list[dict] = []
    after = params["after"]

    while True:
        resp = client.get(f"{BASE_URL}/{endpoint}", params={**params, "after": after}, timeout=30)
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


def fetch(subreddit: str, after: str, before: str, *, resource: str) -> list[dict]:
    with httpx.Client() as client:
        return _paginate(
            client,
            ENDPOINTS[resource],
            {
                "subreddit": subreddit,
                "after": after,
                "before": before,
                "limit": "auto",
                "sort": "asc",
            },
        )


if __name__ == "__main__":
    out = Path(__file__).parent.parent.parent / "tests" / "data"
    out.mkdir(parents=True, exist_ok=True)
    subreddit, after, before = "ciso", "2026-04-01", "2026-04-30"

    for resource in ENDPOINTS:
        records = fetch(subreddit, after, before, resource=resource)
        save_jsonl(records, out / f"r_{subreddit}_{after}_{before}_{resource}.jsonl")
