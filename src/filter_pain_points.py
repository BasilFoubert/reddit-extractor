from src.utils import count_pain_points, filter_pp_by_urgency, load_jsonl, save_jsonl

URGENCY_THRESHOLD = 9

DATASETS = [
    (
        "data/processed/r_ciso_pain_points.jsonl",
        "data/processed/r_ciso_pain_points_filtered.jsonl",
    ),
    (
        "tests/data/small_subreddit_pain_points.jsonl",
        "tests/data/small_subreddit_pain_points_filtered.jsonl",
    ),
]


def main() -> None:
    for input_path, output_path in DATASETS:
        data = load_jsonl(input_path)
        before = count_pain_points(data)
        filter_pp_by_urgency(data, URGENCY_THRESHOLD)
        after = count_pain_points(data)
        save_jsonl(data, output_path)
        print(
            f"{input_path}: {before} pain points → {after} après filtrage (urgency >= {URGENCY_THRESHOLD})"
        )


if __name__ == "__main__":
    main()
