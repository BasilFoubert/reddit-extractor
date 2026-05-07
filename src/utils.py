import json


def load_jsonl(file_path: str)->list[dict]:
    data=[]
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data
