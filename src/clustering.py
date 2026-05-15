import os
from collections import defaultdict

import numpy as np
import torch
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from umap import UMAP

from src.utils import load_jsonl, load_pain_points, save_jsonl


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    pain_points = load_pain_points("data/processed/r_ciso_pain_points.jsonl")
    texts = ["clustering: " + pp["text"] for pp in pain_points]

    # https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device=device
    )
    embeddings = model.encode(texts, show_progress_bar=True)

    np.save("data/embeddings/r_ciso_pain_points_embeddings.npy", embeddings)

    embeddings_umap = UMAP(
        n_components=40, n_neighbors=3, metric="cosine", min_dist=0.0
    ).fit_transform(embeddings)

    np.save("data/embeddings/r_ciso_pain_points_embeddings_umap.npy", embeddings_umap)

    labels = HDBSCAN(min_samples=15, min_cluster_size=20, metric="euclidean").fit_predict(
        embeddings_umap
    )

    save_jsonl(
        [
            {**pp, "cluster": int(label)}
            for pp, label in zip(pain_points, labels, strict=True)
            if label != -1
        ],
        "data/processed/r_ciso_pain_points_clustered.jsonl",
    )

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"{n_clusters} clusters, {int((labels == -1).sum())} noise points")


if __name__ == "__main__":
    if not os.path.exists("data/processed/r_ciso_pain_points_clustered.jsonl"):
        main()

    data = load_jsonl("data/processed/r_ciso_pain_points_clustered.jsonl")
    clusters_map = defaultdict(list)
    for entry in data:
        clusters_map[entry["cluster"]].append(entry["text"])

    print(f"Nombre de clusters trouvés : {len(clusters_map)}")
    print("=" * 50)

    for cluster_id, examples in sorted(clusters_map.items()):
        print(f"\nCLUSTER N°{cluster_id} ({len(examples)} éléments)")
        for i, text in enumerate(examples[:10], 1):
            print(f"  {i}. {text}")
        print("-" * 20)
