import json
from pathlib import Path
from typing import Annotated

JsonlData = list[dict]


def load_jsonl(file_path: str | Path) -> JsonlData:
    data = []
    with open(file_path, encoding="utf-8") as file:
        for line in file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def save_jsonl(data: JsonlData, file_path: str | Path) -> None:
    with open(file_path, "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_pain_points(file_path: str | Path) -> JsonlData:
    data = []
    with open(file_path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                post = json.loads(line)
                for pp in post.get("pain_points", []):
                    data.append(
                        {
                            "text": pp["pain_point_reformulated"],
                            "post_id": pp["post_id"],
                            "verbatim": pp["verbatim"],
                        }
                    )
            except json.JSONDecodeError:
                continue
    return data


def count_pain_points(data: JsonlData) -> int:
    return sum(len(thread.get("pain_points", [])) for thread in data)


def print_pain_points(data: JsonlData) -> None:
    for thread in data:
        post_id = thread.get("post_id") or thread.get("id", "unknown")
        pain_points = thread.get("pain_points", [])
        if not pain_points:
            continue
        print(f"\n[{post_id}]")
        for i, pp in enumerate(pain_points, 1):
            urgency = pp.get("urgency", "?")
            text = pp.get("pain_point_reformulated", "")
            print(f"  {i}. [{urgency}/10] {text}")


def filter_pp_by_urgency(data: JsonlData, urgency_threshold: Annotated[int, "Range 0-10"]):
    "filter pain points by urgency level keep only the pain points above the threshold"
    if not 0 <= urgency_threshold <= 10:
        raise ValueError
    for thread in data:
        thread["pain_points"] = [
            pp for pp in thread["pain_points"] if pp["urgency"] >= urgency_threshold
        ]
