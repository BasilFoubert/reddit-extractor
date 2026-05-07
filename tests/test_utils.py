from src.utils import load_jsonl

def test_load_jsonl():
    path ="tests/data/small_subreddit.jsonl"
    data = load_jsonl(path)
    assert data[0]["id"]=="4tqm40"
