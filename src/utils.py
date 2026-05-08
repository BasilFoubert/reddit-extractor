import json
from pathlib import Path

JsonlData = list[dict]


def load_jsonl(file_path: str | Path) -> JsonlData:
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
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
