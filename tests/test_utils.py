from src.utils import load_jsonl, load_pain_points, filter_pp_by_urgency


def test_load_jsonl():
    path = "tests/data/small_subreddit.jsonl"
    data = load_jsonl(path)
    assert data[0]["id"] == "4tqm40"

def test_load_pain_points():
    path = "tests/data/small_subreddit_pain_points.jsonl"
    data = load_pain_points(path)
    assert len(data) > 0
    assert "text" in data[0]
    assert "post_id" in data[0]
    assert "verbatim" in data[0]

def test_filter_pp_by_urgency():
    path = "tests/data/small_subreddit_pain_points.jsonl"
    data = load_jsonl(path)
    filter_pp_by_urgency(data, 9)
    print(data[1]["pain_points"])
    assert len(data[1]["pain_points"]) == 2
