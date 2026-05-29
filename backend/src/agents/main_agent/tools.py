import pickle
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from src.services.threads_manager import ThreadsManagerService

_TMP_DIR = Path(__file__).parents[3] / "data" / "tmp"


@tool
def list_tmp_files() -> str:
    """List all saved pipeline state files available in the temporary directory.

    Returns:
        A list of pickle filenames with their sizes, or a message if none exist.
    """
    _TMP_DIR.mkdir(parents=True, exist_ok=True)
    pkl_files = sorted(_TMP_DIR.glob("*.pkl"))
    if not pkl_files:
        return "No saved state files found in the temporary directory."
    lines = [f"{f.name} ({f.stat().st_size // 1024} KB)" for f in pkl_files]
    return "Saved state files:\n" + "\n".join(lines)


@tool
def inspect_state(pickle_filename: str) -> str:
    """Inspect the current pipeline state of a saved pickle file.

    Reports what has already been computed: threads, pain points, filtered pain points, clusters.

    Args:
        pickle_filename: filename of the pickle file (e.g. "ciso_2025-01-01_2025-01-31.pkl")

    Returns:
        A summary of what is already available in the state.
    """
    pickle_path = _TMP_DIR / pickle_filename
    if not pickle_path.exists():
        return f"File not found: {pickle_filename}. Use list_tmp_files to see available files."

    with open(pickle_path, "rb") as f:
        svc: ThreadsManagerService = pickle.load(f)

    lines = [f"State for r/{svc.subreddit_name}:"]
    lines.append(f"  threads:        {len(svc.threads)}")
    lines.append(
        f"  pain_points:    {len(svc.pain_points)} {'✓' if svc.pain_points else '✗ not extracted yet'}"
    )
    lines.append(
        f"  filtered_pp:    {len(svc.filtered_pp)} {'✓' if svc.filtered_pp else '✗ not filtered yet'}"
    )
    lines.append(
        f"  clusters:       {len(svc.clusters)} {'✓' if svc.clusters else '✗ not clustered yet'}"
    )
    return "\n".join(lines)


@tool
def extract_pain_points(pickle_filename: str) -> str:
    """Load a saved ThreadsManagerService state from a pickle file and run pain point extraction.

    Args:
        pickle_filename: filename of the pickle file (e.g. "ciso_2025-01-01_2025-01-31.pkl")

    Returns:
        A summary of extracted pain points, and the updated state saved back to the same file.
    """
    pickle_path = _TMP_DIR / pickle_filename
    if not pickle_path.exists():
        return f"File not found: {pickle_filename}. Use list_tmp_files to see available files."

    with open(pickle_path, "rb") as f:
        svc: ThreadsManagerService = pickle.load(f)

    svc.extract_pain_points()

    with open(pickle_path, "wb") as f:
        pickle.dump(svc, f)

    return (
        f"Extracted {len(svc.pain_points)} pain points from {len(svc.threads)} threads "
        f"(r/{svc.subreddit_name}).\n"
        f"State saved to: {pickle_path}"
    )


@tool
def filter_pain_points(pickle_filename: str, urgency_threshold: int = 6) -> str:
    """Load a saved ThreadsManagerService state and filter pain points by urgency score.

    Args:
        pickle_filename: filename of the pickle file (e.g. "ciso_2025-01-01_2025-01-31.pkl")
        urgency_threshold: minimum urgency score to keep (1–10, default 6)

    Returns:
        A summary of filtered pain points, and the updated state saved back to the same file.
    """
    pickle_path = _TMP_DIR / pickle_filename
    if not pickle_path.exists():
        return f"File not found: {pickle_filename}. Use list_tmp_files to see available files."

    with open(pickle_path, "rb") as f:
        svc: ThreadsManagerService = pickle.load(f)

    if not svc.pain_points:
        return "No pain points found in this state. Run run_extract_pain_points first."

    svc.filter_pain_points(urgency_threshold=urgency_threshold)

    with open(pickle_path, "wb") as f:
        pickle.dump(svc, f)

    return (
        f"Filtered to {len(svc.filtered_pp)} pain points (urgency >= {urgency_threshold}) "
        f"out of {len(svc.pain_points)} total (r/{svc.subreddit_name}).\n"
        f"State saved to: {pickle_path}"
    )


@tool
def spot_clusters(pickle_filename: str) -> str:
    """Run hierarchical semantic clustering on filtered pain points from a saved pipeline state.

    Phase 1 groups pain points into tight micro-clusters via LLM-validated k-NN similarity.
    Phase 2 consolidates micro-clusters into broader macro-categories using an LLM.

    Args:
        pickle_filename: filename of the pickle file (e.g. "ciso_2025-01-01_2025-01-31.pkl")

    Returns:
        Confirmation that clustering completed and how many macro-clusters were found.
    """
    pickle_path = _TMP_DIR / pickle_filename
    if not pickle_path.exists():
        return f"File not found: {pickle_filename}. Use list_tmp_files to see available files."

    with open(pickle_path, "rb") as f:
        svc: ThreadsManagerService = pickle.load(f)

    source = svc.filtered_pp if svc.filtered_pp else svc.pain_points
    if not source:
        return "No pain points found in this state. Run extract_pain_points first."

    svc.spot_clusters()

    with open(pickle_path, "wb") as f:
        pickle.dump(svc, f)

    n = len(svc.clusters)
    return (
        f"Clustering complete: {n} macro-cluster{'s' if n != 1 else ''} found "
        f"from {len(source)} pain points (r/{svc.subreddit_name}).\n"
        f"State saved to: {pickle_path}"
    )


@tool
def list_clusters(pickle_filename: str) -> str:
    """List all macro-clusters from a saved pipeline state with their descriptions and sizes.

    Args:
        pickle_filename: filename of the pickle file (e.g. "ciso_2025-01-01_2025-01-31.pkl")

    Returns:
        A numbered list of macro-clusters with description and pain point count for each.
    """
    pickle_path = _TMP_DIR / pickle_filename
    if not pickle_path.exists():
        return f"File not found: {pickle_filename}. Use list_tmp_files to see available files."

    with open(pickle_path, "rb") as f:
        svc: ThreadsManagerService = pickle.load(f)

    if not svc.clusters:
        return "No clusters found in this state. Run spot_clusters first."

    lines = [f"{len(svc.clusters)} macro-clusters (r/{svc.subreddit_name}):\n"]
    for i, cluster in enumerate(svc.clusters, 1):
        lines.append(f"{i}. {cluster['description']} — {len(cluster['pain_points'])} pain points")
    return "\n".join(lines)


@tool
def get_cluster(pickle_filename: str, cluster_index: int) -> str:
    """Get the full details of a specific macro-cluster, including all its pain points.

    Args:
        pickle_filename: filename of the pickle file (e.g. "ciso_2025-01-01_2025-01-31.pkl")
        cluster_index: 1-based index of the cluster (as shown by list_clusters)

    Returns:
        The cluster description and all its pain points with verbatim quotes and urgency scores.
    """
    pickle_path = _TMP_DIR / pickle_filename
    if not pickle_path.exists():
        return f"File not found: {pickle_filename}. Use list_tmp_files to see available files."

    with open(pickle_path, "rb") as f:
        svc: ThreadsManagerService = pickle.load(f)

    if not svc.clusters:
        return "No clusters found in this state. Run spot_clusters first."

    if cluster_index < 1 or cluster_index > len(svc.clusters):
        return f"Invalid index {cluster_index}. There are {len(svc.clusters)} clusters (1–{len(svc.clusters)})."

    cluster = svc.clusters[cluster_index - 1]
    lines = [
        f"Cluster {cluster_index}: {cluster['description']}",
        f"{len(cluster['pain_points'])} pain points:\n",
    ]
    for i, pp in enumerate(cluster["pain_points"], 1):
        lines.append(f"{i}. [{pp.urgency}/10] {pp.pain_point_reformulated}")
        lines.append(f'   "{pp.verbatim}"')
    return "\n".join(lines)


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
