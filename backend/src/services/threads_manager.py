class ThreadsManagerService:
    def __init__(self, subreddit_name: str):
        self.subreddit_name = subreddit_name

    def download_subreddit(self):
        """Download subreddit data from Artic Shift"""
        pass

    def ingest_posts(self) -> int:
        """Normalize raw posts. Returns the number of posts."""
        pass

    def ingest_comments(self) -> int:
        """Filter and normalize comments. Returns the number of comments."""
        pass

    def build_threads(self) -> int:
        """Assemble posts and comments into threads. Returns the number of threads."""
        pass

    def extract_pain_points(self) -> int:
        """Extract pain points from threads using LLM. Returns the number of pain points."""
        pass

    def filter_pain_points(self) -> int:
        """Filter pain points by urgency threshold. Returns the number of pain points kept."""
        pass

    def run_pipeline(self) -> None:
        """Run all pipeline steps in order."""
        pass
