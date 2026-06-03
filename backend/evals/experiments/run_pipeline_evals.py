from pathlib import Path

from dotenv import load_dotenv
from evals.evaluators.evaluators import cluster_assignment_judge
from langsmith import evaluate

load_dotenv(Path(__file__).parents[3] / ".env")

_DATASET_NAME = "ciso_2026-04-01_2026-04-30"

if __name__ == "__main__":
    results = evaluate(
        lambda inputs: inputs,
        data=_DATASET_NAME,
        evaluators=[cluster_assignment_judge],
        experiment_prefix="cluster-accuracy",
    )
    scores = [r["evaluation_results"]["results"][0].score for r in results]
    print(f"Score moyen : {sum(scores) / len(scores):.2%}")
    print(f"Bien classés : {sum(scores)}/{len(scores)}")
