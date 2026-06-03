import pickle
from pathlib import Path

from dotenv import load_dotenv

from src.services.threads_manager import ThreadsManagerService

load_dotenv(Path(__file__).parents[2] / ".env")

_PKL = Path(__file__).parent / "ciso_2026-04-01_2026-04-30.pkl"

if __name__ == "__main__":
    with open(_PKL, "rb") as f:
        svc: ThreadsManagerService = pickle.load(f)

    svc.stats()
