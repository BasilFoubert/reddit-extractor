import pickle
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

from src.services.threads_manager import ThreadsManagerService

load_dotenv(Path(__file__).parents[3] / ".env")

_HERE = Path(__file__).parent
_DATASET_NAME = "ciso_2026-04-01_2026-04-30"
_PKL = _HERE / "ciso_2026-04-01_2026-04-30.pkl"
_ORG_ID = "a57b71fa-5b63-452b-a696-3a5b9f0b7550"

if __name__ == "__main__":
    with open(_PKL, "rb") as f:
        svc: ThreadsManagerService = pickle.load(f)

    client = Client()
    flatten_dataset = svc.flatten_dataset()

    client.delete_dataset(dataset_name=_DATASET_NAME)
    dataset = client.create_dataset(_DATASET_NAME)
    client.create_examples(inputs=flatten_dataset, dataset_id=dataset.id)

    print(f"Dataset '{_DATASET_NAME}' created with {len(flatten_dataset)} examples.")
    print(f"https://smith.langchain.com/o/{_ORG_ID}/datasets/{dataset.id}")