import pickle
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from src.services.threads_manager import ThreadsManagerService

_TMP_DIR = Path(__file__).parents[2] / "data" / "tmp"


@tool
def build_user_thread(subreddit: str, start_date: str, end_date: str) -> str:
    """Download and preprocess Reddit threads for a subreddit over a date range.
    Runs download, ingest, and thread-building stages, then saves state to a pickle file
    so subsequent tools can continue the pipeline.

    Args:
        subreddit: subreddit name without r/ prefix (e.g. "ciso")
        start_date: start date in YYYY-MM-DD format
        end_date: end date in YYYY-MM-DD format

    Returns:
        Path to the pickle file and a summary of what was collected.
    """
    _TMP_DIR.mkdir(parents=True, exist_ok=True)

    after = str(int(datetime.strptime(start_date, "%Y-%m-%d").timestamp()))
    before = str(int(datetime.strptime(end_date, "%Y-%m-%d").timestamp()))

    svc = ThreadsManagerService()
    svc.set_subreddit_name(subreddit)
    svc.set_start_date(after)
    svc.set_end_date(before)

    svc.download_subreddit(after, before)
    svc.ingest_posts()
    svc.ingest_comments()
    svc.build_threads()

    pickle_path = _TMP_DIR / f"{subreddit}_{start_date}_{end_date}.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(svc, f)

    return (
        f"Built {len(svc.threads)} threads from r/{subreddit} ({start_date} → {end_date}).\n"
        f"State saved to: {pickle_path}"
    )
